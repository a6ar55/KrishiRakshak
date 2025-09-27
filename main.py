"""
Main pipeline script for Plant Disease Segmentation System
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.config import Config, parse_args, override_config_from_args, setup_seed, create_experiment_dir, validate_config
from src.training.trainer import PlantDiseaseTrainer
from src.inference.predictor import PlantDiseasePredictor
from src.data.dataset import create_data_splits
from src.utils.visualization import create_segmentation_report


def train_mode(config: Config, args: argparse.Namespace):
    """Training mode"""
    print("=" * 60)
    print("PLANT DISEASE SEGMENTATION - TRAINING MODE")
    print("=" * 60)
    
    # Setup reproducibility
    setup_seed(config)
    
    # Validate configuration
    validate_config(config)
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config, args.experiment_name)
    
    # Initialize trainer
    trainer = PlantDiseaseTrainer(config, experiment_dir)
    
    # Start training
    trainer.train()
    
    print(f"\nTraining completed! Results saved to: {experiment_dir}")


def inference_mode(config: Config, args: argparse.Namespace):
    """Inference mode"""
    print("=" * 60)
    print("PLANT DISEASE SEGMENTATION - INFERENCE MODE")
    print("=" * 60)
    
    if not args.model_path:
        raise ValueError("Model path is required for inference mode")
    
    if not args.input_path:
        raise ValueError("Input path is required for inference mode")
    
    # Initialize predictor
    predictor = PlantDiseasePredictor(
        model_path=args.model_path,
        config_path=args.config,
        device=config.get('hardware.device', 'auto'),
        image_size=config.get('data.image_size', [512, 512])
    )
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else Path('inference_results')
    
    # Process input
    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        result = predictor.predict_single_image(input_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        base_name = input_path.stem
        
        # Save mask
        import cv2
        mask_path = output_path / f'{base_name}_mask.png'
        cv2.imwrite(str(mask_path), result['prediction_mask'] * 255)
        
        # Save visualization
        vis_path = output_path / f'{base_name}_result.png'
        predictor._save_visualization(result, vis_path)
        
        print(f"\nResults:")
        print(f"Disease coverage: {result['disease_percentage']:.2f}%")
        print(f"Diseased pixels: {result['diseased_pixels']:,}")
        print(f"Total pixels: {result['total_pixels']:,}")
        print(f"Results saved to: {output_path}")
        
    elif input_path.is_dir():
        # Directory inference
        print(f"Processing directory: {input_path}")
        results = predictor.predict_directory(
            input_dir=input_path,
            output_dir=output_path,
            save_visualizations=True,
            save_masks=True
        )
        
        # Print summary statistics
        if results:
            disease_percentages = [r['disease_percentage'] for r in results]
            print(f"\nSummary Statistics:")
            print(f"Images processed: {len(results)}")
            print(f"Average disease coverage: {sum(disease_percentages)/len(disease_percentages):.2f}%")
            print(f"Min disease coverage: {min(disease_percentages):.2f}%")
            print(f"Max disease coverage: {max(disease_percentages):.2f}%")
    
    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


def evaluate_mode(config: Config, args: argparse.Namespace):
    """Evaluation mode"""
    print("=" * 60)
    print("PLANT DISEASE SEGMENTATION - EVALUATION MODE")
    print("=" * 60)
    
    if not args.model_path:
        raise ValueError("Model path is required for evaluation mode")
    
    # Setup data module for testing
    from src.data.dataset import PlantDiseaseDataModule
    
    data_module = PlantDiseaseDataModule(
        data_dir=config.get('data.data_dir'),
        batch_size=config.get('inference.batch_size', 8),
        image_size=config.get('data.image_size', [512, 512]),
        num_workers=config.get('data.num_workers', 4)
    )
    
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    # Load model
    device = torch.device(config.get('hardware.device', 'cuda') if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    from src.models.enet import create_enet_model
    model = create_enet_model(num_classes=config.get('model.num_classes', 2))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create evaluation report
    output_dir = Path(args.output_path) if args.output_path else Path('evaluation_results')
    create_segmentation_report(
        model=model,
        test_loader=test_loader,
        device=str(device),
        save_dir=str(output_dir),
        num_samples=10
    )
    
    print(f"Evaluation completed! Results saved to: {output_dir}")


def data_preparation_mode(config: Config, args: argparse.Namespace):
    """Data preparation mode"""
    print("=" * 60)
    print("PLANT DISEASE SEGMENTATION - DATA PREPARATION MODE")
    print("=" * 60)
    
    if not args.source_data_dir:
        raise ValueError("Source data directory is required for data preparation mode")
    
    source_dir = args.source_data_dir
    target_dir = config.get('data.data_dir', 'data')
    
    print(f"Creating data splits from: {source_dir}")
    print(f"Target directory: {target_dir}")
    
    create_data_splits(
        source_dir=source_dir,
        target_dir=target_dir,
        train_ratio=args.train_ratio or 0.7,
        val_ratio=args.val_ratio or 0.2,
        test_ratio=args.test_ratio or 0.1
    )
    
    print("Data preparation completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Plant Disease Segmentation System")
    
    # Mode selection
    parser.add_argument('mode', choices=['train', 'inference', 'evaluate', 'prepare_data'],
                       help='Mode to run the system in')
    
    # Configuration
    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    
    # Common arguments
    parser.add_argument('--experiment_name', type=str, default='plant_disease_segmentation',
                       help='Name of the experiment')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    # Training arguments
    parser.add_argument('--data_dir', type=str,
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', '--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    # Inference/Evaluation arguments
    parser.add_argument('--model_path', '-m', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_path', '-i', type=str,
                       help='Path to input image or directory')
    parser.add_argument('--output_path', '-o', type=str,
                       help='Path to output directory')
    
    # Data preparation arguments
    parser.add_argument('--source_data_dir', type=str,
                       help='Source directory containing raw data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        config = Config()
        # Set default values
        config.set('model.num_classes', 2)
        config.set('data.data_dir', 'data')
        config.set('data.batch_size', 16)
        config.set('training.epochs', 100)
        config.set('training.learning_rate', 0.001)
    
    # Override config from command line arguments
    if args.data_dir:
        config.set('data.data_dir', args.data_dir)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    if args.device:
        config.set('hardware.device', args.device)
    if args.resume:
        config.set('checkpoint.resume_from', args.resume)
    
    if args.debug:
        config.set('training.epochs', 5)
        config.set('data.batch_size', 4)
        print("Debug mode enabled")
    
    # Run selected mode
    try:
        if args.mode == 'train':
            train_mode(config, args)
        elif args.mode == 'inference':
            inference_mode(config, args)
        elif args.mode == 'evaluate':
            evaluate_mode(config, args)
        elif args.mode == 'prepare_data':
            data_preparation_mode(config, args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Inference script for plant disease segmentation
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from src.models.enet import create_enet_model
from src.utils.config import Config
from src.utils.visualization import overlay_mask_on_image, create_color_mask


class PlantDiseasePredictor:
    """
    Predictor class for plant disease segmentation inference
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'auto',
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to trained model checkpoint
            config_path (str, optional): Path to configuration file
            device (str): Device to run inference on
            image_size (tuple): Input image size for model
        """
        self.model_path = model_path
        self.image_size = image_size
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load configuration if provided
        self.config = None
        if config_path:
            self.config = Config(config_path)
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessing transforms
        self.transform = self._create_transforms()
        
        print("Predictor initialized successfully")
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if self.config:
            num_classes = self.config.get('model.num_classes', 2)
        else:
            # Try to get from checkpoint config
            if 'config' in checkpoint:
                num_classes = checkpoint['config'].get('model', {}).get('num_classes', 2)
            else:
                num_classes = 2  # Default for binary segmentation
        
        # Create model
        model = create_enet_model(num_classes=num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {self.model_path}")
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        
        return model
    
    def _create_transforms(self) -> A.Compose:
        """Create preprocessing transforms"""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image (np.ndarray): Input image in RGB format
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_prediction(
        self,
        prediction: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Postprocess model prediction
        
        Args:
            prediction (torch.Tensor): Model output [1, C, H, W] or [1, H, W]
            original_size (tuple): Original image size (height, width)
        
        Returns:
            np.ndarray: Segmentation mask resized to original size
        """
        # Convert to numpy
        if prediction.dim() == 4:  # [1, C, H, W]
            prediction = torch.argmax(prediction, dim=1)  # [1, H, W]
        
        prediction = prediction.squeeze().cpu().numpy()  # [H, W]
        
        # Resize to original size
        prediction_resized = cv2.resize(
            prediction.astype(np.uint8),
            (original_size[1], original_size[0]),  # (width, height)
            interpolation=cv2.INTER_NEAREST
        )
        
        return prediction_resized
    
    def predict_single_image(
        self,
        image_path: Union[str, Path],
        return_probabilities: bool = False
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Predict segmentation for a single image
        
        Args:
            image_path (Union[str, Path]): Path to input image
            return_probabilities (bool): Whether to return class probabilities
        
        Returns:
            Dict containing prediction results
        """
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if return_probabilities:
                probabilities = F.softmax(output, dim=1).squeeze().cpu().numpy()
            else:
                probabilities = None
        
        # Postprocess
        prediction_mask = self.postprocess_prediction(output, original_size)
        
        # Calculate disease statistics
        total_pixels = prediction_mask.size
        diseased_pixels = (prediction_mask == 1).sum()
        disease_percentage = (diseased_pixels / total_pixels) * 100
        
        # Create colored mask
        colored_mask = create_color_mask(prediction_mask)
        
        # Create overlay
        overlay = overlay_mask_on_image(image, prediction_mask, alpha=0.4)
        
        results = {
            'original_image': image,
            'prediction_mask': prediction_mask,
            'colored_mask': colored_mask,
            'overlay': overlay,
            'disease_percentage': disease_percentage,
            'diseased_pixels': diseased_pixels,
            'total_pixels': total_pixels
        }
        
        if probabilities is not None:
            # Resize probabilities to original size
            prob_resized = []
            for i in range(probabilities.shape[0]):
                prob_class = cv2.resize(
                    probabilities[i],
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                prob_resized.append(prob_class)
            results['probabilities'] = np.stack(prob_resized)
        
        return results
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Predict segmentation for multiple images in batches
        
        Args:
            image_paths (List): List of image paths
            batch_size (int): Batch size for inference
        
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            for image_path in batch_paths:
                try:
                    result = self.predict_single_image(image_path)
                    result['image_path'] = str(image_path)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        return results
    
    def predict_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualizations: bool = True,
        save_masks: bool = True,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Predict segmentation for all images in a directory
        
        Args:
            input_dir (Union[str, Path]): Input directory containing images
            output_dir (Union[str, Path]): Output directory for results
            save_visualizations (bool): Whether to save visualization images
            save_masks (bool): Whether to save segmentation masks
            image_extensions (List[str]): Supported image extensions
        
        Returns:
            List of prediction results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            (output_dir / 'visualizations').mkdir(exist_ok=True)
        if save_masks:
            (output_dir / 'masks').mkdir(exist_ok=True)
        
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f'*{ext}'))
            image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images in {input_dir}")
        
        # Process images
        results = []
        disease_stats = []
        
        for image_path in image_paths:
            try:
                print(f"Processing: {image_path.name}")
                result = self.predict_single_image(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
                
                # Save outputs
                base_name = image_path.stem
                
                if save_masks:
                    mask_path = output_dir / 'masks' / f'{base_name}_mask.png'
                    cv2.imwrite(str(mask_path), result['prediction_mask'] * 255)
                
                if save_visualizations:
                    vis_path = output_dir / 'visualizations' / f'{base_name}_result.png'
                    self._save_visualization(result, vis_path)
                
                # Collect statistics
                disease_stats.append({
                    'image': image_path.name,
                    'disease_percentage': result['disease_percentage'],
                    'diseased_pixels': result['diseased_pixels'],
                    'total_pixels': result['total_pixels']
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Save statistics
        self._save_statistics(disease_stats, output_dir / 'statistics.txt')
        
        print(f"\nProcessing completed!")
        print(f"Results saved to: {output_dir}")
        print(f"Successfully processed: {len(results)}/{len(image_paths)} images")
        
        return results
    
    def _save_visualization(self, result: Dict, save_path: Path):
        """Save visualization of prediction result"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(result['colored_mask'])
        axes[1].set_title('Disease Segmentation')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(result['overlay'])
        axes[2].set_title(f'Overlay (Disease: {result["disease_percentage"]:.1f}%)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_statistics(self, stats: List[Dict], save_path: Path):
        """Save disease statistics to file"""
        with open(save_path, 'w') as f:
            f.write("Plant Disease Segmentation Results\n")
            f.write("=" * 50 + "\n\n")
            
            total_images = len(stats)
            avg_disease_percentage = np.mean([s['disease_percentage'] for s in stats])
            
            f.write(f"Total Images Processed: {total_images}\n")
            f.write(f"Average Disease Coverage: {avg_disease_percentage:.2f}%\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 50 + "\n")
            
            for stat in stats:
                f.write(f"Image: {stat['image']}\n")
                f.write(f"  Disease Coverage: {stat['disease_percentage']:.2f}%\n")
                f.write(f"  Diseased Pixels: {stat['diseased_pixels']:,}\n")
                f.write(f"  Total Pixels: {stat['total_pixels']:,}\n\n")


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plant Disease Segmentation Inference")
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='Input image size (height width)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing multiple images')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save segmentation masks')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PlantDiseasePredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        image_size=tuple(args.image_size)
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Process input
    if input_path.is_file():
        # Single image
        print(f"Processing single image: {input_path}")
        result = predictor.predict_single_image(input_path)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        base_name = input_path.stem
        
        if args.save_masks:
            mask_path = output_path / f'{base_name}_mask.png'
            cv2.imwrite(str(mask_path), result['prediction_mask'] * 255)
            print(f"Mask saved to: {mask_path}")
        
        if args.save_visualizations:
            vis_path = output_path / f'{base_name}_result.png'
            predictor._save_visualization(result, vis_path)
            print(f"Visualization saved to: {vis_path}")
        
        print(f"Disease coverage: {result['disease_percentage']:.2f}%")
        
    elif input_path.is_dir():
        # Directory of images
        print(f"Processing directory: {input_path}")
        results = predictor.predict_directory(
            input_dir=input_path,
            output_dir=output_path,
            save_visualizations=args.save_visualizations,
            save_masks=args.save_masks
        )
        
        # Print summary
        if results:
            avg_disease = np.mean([r['disease_percentage'] for r in results])
            print(f"Average disease coverage: {avg_disease:.2f}%")
        
    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


if __name__ == "__main__":
    main()


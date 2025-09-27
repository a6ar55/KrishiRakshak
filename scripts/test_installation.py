"""
Test script to verify installation and basic functionality
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.models.enet import create_enet_model
        from src.data.dataset import PlantDiseaseDataset, PlantDiseaseDataModule
        from src.utils.metrics import SegmentationMetrics, dice_loss, focal_loss
        from src.utils.config import Config
        from src.utils.visualization import plot_sample_batch
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_model():
    """Test model creation and forward pass"""
    print("Testing model...")
    
    try:
        model = create_enet_model(num_classes=2)
        
        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 2, 512, 512), f"Expected shape (1, 2, 512, 512), got {output.shape}"
        
        print(f"✓ Model created successfully")
        print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False


def test_metrics():
    """Test metrics computation"""
    print("Testing metrics...")
    
    try:
        from src.utils.metrics import SegmentationMetrics, dice_loss, focal_loss
        
        # Create dummy data
        predictions = torch.randn(4, 2, 256, 256)
        targets = torch.randint(0, 2, (4, 256, 256))
        
        # Test metrics
        metrics = SegmentationMetrics(num_classes=2)
        metrics.update(predictions, targets)
        results = metrics.compute_all_metrics()
        
        assert 'mean_iou' in results
        assert 'mean_dice' in results
        assert 'pixel_accuracy' in results
        
        # Test loss functions
        dice_loss_val = dice_loss(predictions, targets)
        focal_loss_val = focal_loss(predictions, targets)
        
        assert isinstance(dice_loss_val.item(), float)
        assert isinstance(focal_loss_val.item(), float)
        
        print("✓ Metrics computation successful")
        print(f"✓ Sample IoU: {results['mean_iou']:.4f}")
        print(f"✓ Sample Dice: {results['mean_dice']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Metrics test error: {e}")
        return False


def test_config():
    """Test configuration system"""
    print("Testing configuration...")
    
    try:
        from src.utils.config import Config
        
        # Test default config
        config = Config()
        config.set('test.value', 42)
        assert config.get('test.value') == 42
        
        # Test nested config
        config.set('model.num_classes', 2)
        assert config.get('model.num_classes') == 2
        
        print("✓ Configuration system working")
        return True
    except Exception as e:
        print(f"✗ Config test error: {e}")
        return False


def test_device():
    """Test device availability"""
    print("Testing device availability...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    if device.type == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name()}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True


def main():
    """Main test function"""
    print("Plant Disease Segmentation - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model", test_model),
        ("Metrics", test_metrics),
        ("Configuration", test_config),
        ("Device", test_device)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Prepare your dataset using: python scripts/download_sample_data.py")
        print("2. Start training: python main.py train")
        print("3. Run inference: python main.py inference --model_path checkpoints/best.pth --input_path image.jpg")
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)


if __name__ == "__main__":
    main()



"""
Train ENet model on Bacterial Spot dataset for disease segmentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import random
import sys

# Add src to path
sys.path.append('src')
from src.models.simple_unet import create_simple_unet


class BacterialSpotDataset(Dataset):
    """Dataset for Bacterial Spot segmentation"""
    
    def __init__(self, image_paths, label_paths, image_size=(512, 512), augment=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_size = image_size
        self.augment = augment
        
        # Create transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = self.label_paths[idx]
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        
        # Convert label to binary (0: background, 1: diseased)
        label = (label > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # Ensure label is long tensor
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        else:
            label = label.long()
        
        return {
            'image': image,
            'mask': label,
            'image_path': str(image_path),
            'label_path': str(label_path)
        }


def create_dataset_splits(images_dir, labels_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Create train/val/test splits"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
    
    # Find corresponding label files
    matched_pairs = []
    for image_file in image_files:
        # Create expected label filename
        label_name = f"{image_file.stem}_label.png"
        label_file = labels_dir / label_name
        
        if label_file.exists():
            matched_pairs.append((image_file, label_file))
        else:
            print(f"Warning: No label found for {image_file.name}")
    
    print(f"Found {len(matched_pairs)} matching image-label pairs")
    
    if len(matched_pairs) == 0:
        raise ValueError("No matching image-label pairs found!")
    
    # Split the data
    image_paths, label_paths = zip(*matched_pairs)
    
    # First split: train and temp (val + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_paths, label_paths, test_size=(1 - train_ratio), random_state=42
    )
    
    # Second split: val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=(1 - val_size), random_state=42
    )
    
    print(f"Train: {len(train_images)} pairs")
    print(f"Validation: {len(val_images)} pairs")
    print(f"Test: {len(test_images)} pairs")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def dice_loss(predictions, targets, smooth=1e-8):
    """Compute Dice loss"""
    predictions = F.softmax(predictions, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
    
    predictions = predictions.view(predictions.shape[0], predictions.shape[1], -1)
    targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
    
    intersection = (predictions * targets_one_hot).sum(dim=2)
    dice = (2 * intersection + smooth) / (predictions.sum(dim=2) + targets_one_hot.sum(dim=2) + smooth)
    
    return 1 - dice.mean()


def combined_loss(predictions, targets, ce_weight=0.7, dice_weight=0.3):
    """Combined loss function"""
    ce_loss = F.cross_entropy(predictions, targets)
    dice_loss_val = dice_loss(predictions, targets)
    return ce_weight * ce_loss + dice_weight * dice_loss_val


def calculate_iou(predictions, targets, num_classes=2):
    """Calculate IoU for each class"""
    ious = []
    predictions = torch.argmax(predictions, dim=1)
    
    for class_id in range(num_classes):
        pred_class = (predictions == class_id)
        target_class = (targets == class_id)
        
        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()
        
        if union == 0:
            iou = 1.0  # Perfect score if no pixels of this class
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return ious


def train_model():
    """Train the ENet model"""
    print("=" * 60)
    print("TRAINING ENET ON BACTERIAL SPOT DATASET")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset splits
    images_dir = "Bacterial Spot/Bacterial Spot"
    labels_dir = "Bacterial_labels"
    
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = create_dataset_splits(
        images_dir, labels_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )
    
    # Create datasets
    train_dataset = BacterialSpotDataset(train_images, train_labels, image_size=(256, 256), augment=True)
    val_dataset = BacterialSpotDataset(val_images, val_labels, image_size=(256, 256), augment=False)
    test_dataset = BacterialSpotDataset(test_images, test_labels, image_size=(256, 256), augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Create model (binary segmentation: background vs diseased)
    model = create_simple_unet(num_classes=2)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    num_epochs = 20
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
        
        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_val_ious = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                # Calculate IoU
                batch_ious = calculate_iou(outputs, masks)
                all_val_ious.append(batch_ious)
        
        avg_val_loss = val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        # Calculate mean IoU
        mean_background_iou = np.mean([ious[0] for ious in all_val_ious])
        mean_diseased_iou = np.mean([ious[1] for ious in all_val_ious])
        mean_iou = (mean_background_iou + mean_diseased_iou) / 2
        val_ious.append(mean_iou)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU - Background: {mean_background_iou:.4f}, Diseased: {mean_diseased_iou:.4f}, Mean: {mean_iou:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': mean_iou,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, 'unet_bacterial_spot_best.pth')
            print(f"✓ Best model saved! IoU: {best_val_iou:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_ious, label='Validation IoU')
    plt.title('Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    lr_history = [0.001 * (0.5 ** (i // 5)) for i in range(len(train_losses))]  # Approximate LR schedule
    plt.plot(lr_history, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('enet_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTraining completed!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    
    return model, test_loader, (test_images, test_labels)


def test_model_on_random_images(model, test_loader, test_data, device, num_samples=5):
    """Test the trained model on random images"""
    print("\n" + "=" * 60)
    print("TESTING MODEL ON RANDOM IMAGES")
    print("=" * 60)
    
    model.eval()
    
    # Get random test samples
    test_images, test_labels = test_data
    random_indices = random.sample(range(len(test_images)), min(num_samples, len(test_images)))
    
    # Create results directory
    results_dir = Path("bacterial_spot_results")
    results_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            # Load original image and label
            image_path = test_images[idx]
            label_path = test_labels[idx]
            
            # Load and preprocess
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            original_label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            original_label = (original_label > 127).astype(np.uint8)
            
            # Prepare for model input
            transform = A.Compose([
                A.Resize(height=256, width=256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            transformed = transform(image=original_image, mask=original_label)
            input_image = transformed['image'].unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_image)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8),
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Calculate metrics
            intersection = (prediction_resized & original_label).sum()
            union = (prediction_resized | original_label).sum()
            iou = intersection / union if union > 0 else 1.0
            
            diseased_pixels = prediction_resized.sum()
            total_pixels = prediction_resized.size
            disease_percentage = (diseased_pixels / total_pixels) * 100
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title(f'Original Image\n{Path(image_path).name}')
            axes[0, 0].axis('off')
            
            # Ground truth
            axes[0, 1].imshow(original_label, cmap='Reds', alpha=0.7)
            axes[0, 1].imshow(original_image, alpha=0.3)
            axes[0, 1].set_title('Ground Truth Overlay')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[1, 0].imshow(prediction_resized, cmap='Reds', alpha=0.7)
            axes[1, 0].imshow(original_image, alpha=0.3)
            axes[1, 0].set_title(f'Prediction Overlay\nDisease: {disease_percentage:.1f}%')
            axes[1, 0].axis('off')
            
            # Comparison
            comparison = np.zeros((*original_image.shape[:2], 3), dtype=np.uint8)
            comparison[original_label == 1] = [0, 255, 0]  # Ground truth in green
            comparison[prediction_resized == 1] = [255, 0, 0]  # Prediction in red
            comparison[(original_label == 1) & (prediction_resized == 1)] = [255, 255, 0]  # Overlap in yellow
            
            axes[1, 1].imshow(comparison)
            axes[1, 1].set_title(f'Comparison\nIoU: {iou:.3f}\nGreen=GT, Red=Pred, Yellow=Overlap')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'test_result_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Sample {i+1}:")
            print(f"  Image: {Path(image_path).name}")
            print(f"  Disease coverage: {disease_percentage:.2f}%")
            print(f"  IoU: {iou:.3f}")
            print()
    
    print(f"Test results saved to: {results_dir}")


def main():
    """Main function"""
    try:
        # Train the model
        print("Starting ENet training on Bacterial Spot dataset...")
        model, test_loader, test_data = train_model()
        
        # Test on random images
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_model_on_random_images(model, test_loader, test_data, device, num_samples=5)
        
        print("\n" + "=" * 60)
        print("🎉 BACTERIAL SPOT SEGMENTATION COMPLETED!")
        print("=" * 60)
        print("✓ Model trained successfully")
        print("✓ Best model saved as 'unet_bacterial_spot_best.pth'")
        print("✓ Test results saved in 'bacterial_spot_results/' directory")
        print("✓ Training history saved as 'enet_training_history.png'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

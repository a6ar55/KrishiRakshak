"""
Simple training and testing script for plant disease segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add src to path
sys.path.append('src')
from src.data.dataset import PlantDiseaseDataset


class SimpleUNet(nn.Module):
    """Simple U-Net for plant disease segmentation"""
    
    def __init__(self, num_classes=3):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        
        c3 = self.conv3(p2)
        
        # Decoder
        up2 = self.up2(c3)
        merge2 = torch.cat([up2, c2], dim=1)
        c_up2 = self.conv_up2(merge2)
        
        up1 = self.up1(c_up2)
        merge1 = torch.cat([up1, c1], dim=1)
        c_up1 = self.conv_up1(merge1)
        
        out = self.final(c_up1)
        return out


def train_model():
    """Train the segmentation model"""
    print("=" * 60)
    print("SIMPLE PLANT DISEASE SEGMENTATION TRAINING")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cpu')  # Using CPU for compatibility
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleUNet(num_classes=3)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data
    data_dir = "data/disease_segmentation"
    
    train_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        split='train',
        image_size=(256, 256),
        normalize=True
    )
    
    val_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        split='val',
        image_size=(256, 256),
        normalize=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 3
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += masks.numel()
            train_correct += (predicted == masks).sum().item()
            
            if batch_idx >= 10:  # Limit training for demo
                break
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += masks.numel()
                val_correct += (predicted == masks).sum().item()
                
                if batch_idx >= 5:  # Limit validation for demo
                    break
        
        # Print epoch results
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Train Loss: {train_loss/11:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/6:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'simple_model_best.pth')
            print("✓ Best model saved!")
    
    print("\nTraining completed!")
    return model


def test_model():
    """Test the trained model on sample images"""
    print("\n" + "=" * 60)
    print("TESTING TRAINED MODEL")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Load model
    model = SimpleUNet(num_classes=3)
    checkpoint = torch.load('simple_model_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Load test data
    data_dir = "data/disease_segmentation"
    test_dataset = PlantDiseaseDataset(
        data_dir=data_dir,
        split='test',
        image_size=(256, 256),
        normalize=True
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Create results directory
    results_dir = Path("simple_results")
    results_dir.mkdir(exist_ok=True)
    
    # Test on first few samples
    num_test_samples = min(5, len(test_dataset))
    
    colors = np.array([
        [0, 0, 0],      # Background - Black
        [0, 255, 0],    # Healthy - Green
        [255, 0, 0]     # Diseased - Red
    ])
    
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_test_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get prediction
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            
            # Convert to numpy for visualization
            image = images[0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize image
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)
            
            true_mask = masks[0].cpu().numpy()
            pred_mask = predicted[0].cpu().numpy()
            
            # Create colored masks
            true_colored = colors[true_mask]
            pred_colored = colors[pred_mask]
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Ground truth
            axes[0, 1].imshow(true_colored)
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')
            
            # Prediction
            axes[1, 0].imshow(pred_colored)
            axes[1, 0].set_title('Prediction')
            axes[1, 0].axis('off')
            
            # Overlay
            overlay = (image * 255).astype(np.uint8)
            overlay_with_pred = cv2.addWeighted(overlay, 0.7, pred_colored.astype(np.uint8), 0.3, 0)
            axes[1, 1].imshow(overlay_with_pred)
            axes[1, 1].set_title('Prediction Overlay')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'test_result_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate disease percentage
            total_pixels = pred_mask.size
            diseased_pixels = (pred_mask == 2).sum()
            disease_percentage = (diseased_pixels / total_pixels) * 100
            
            print(f"Sample {idx+1}: Disease coverage = {disease_percentage:.2f}%")
    
    print(f"\nTest results saved to: {results_dir}")
    print("✓ Testing completed successfully!")


def main():
    """Main function"""
    try:
        # Train the model
        model = train_model()
        
        # Test the model
        test_model()
        
        print("\n" + "=" * 60)
        print("🎉 PLANT DISEASE SEGMENTATION DEMO COMPLETED!")
        print("=" * 60)
        print("Results are saved in 'simple_results/' directory")
        print("Trained model is saved as 'simple_model_best.pth'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

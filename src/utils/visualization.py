"""
Visualization utilities for plant disease segmentation
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path
import matplotlib.patches as patches


def denormalize_image(
    image: torch.Tensor, 
    mean: List[float] = [0.485, 0.456, 0.406], 
    std: List[float] = [0.229, 0.224, 0.225]
) -> np.ndarray:
    """
    Denormalize image tensor for visualization
    
    Args:
        image (torch.Tensor): Normalized image tensor [C, H, W]
        mean (List[float]): Normalization mean
        std (List[float]): Normalization std
    
    Returns:
        np.ndarray: Denormalized image [H, W, C]
    """
    if isinstance(image, torch.Tensor):
        image = image.clone()
        
        # Denormalize
        for i in range(len(mean)):
            image[i] = image[i] * std[i] + mean[i]
        
        # Convert to numpy and transpose
        image = image.cpu().numpy().transpose(1, 2, 0)
    
    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def create_color_mask(
    mask: np.ndarray, 
    num_classes: int = 2, 
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create colored mask for visualization
    
    Args:
        mask (np.ndarray): Segmentation mask [H, W]
        num_classes (int): Number of classes
        colors (List[Tuple]): RGB colors for each class
    
    Returns:
        np.ndarray: Colored mask [H, W, 3]
    """
    if colors is None:
        # Default colors: black for healthy, red for diseased
        colors = [
            (0, 0, 0),        # Class 0: Healthy (black)
            (255, 0, 0),      # Class 1: Diseased (red)
        ]
        
        # Add more colors if needed
        if num_classes > 2:
            additional_colors = [
                (0, 255, 0),      # Green
                (0, 0, 255),      # Blue
                (255, 255, 0),    # Yellow
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Cyan
            ]
            colors.extend(additional_colors[:num_classes-2])
    
    # Create colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = colors[class_id]
    
    return colored_mask


def overlay_mask_on_image(
    image: np.ndarray, 
    mask: np.ndarray, 
    alpha: float = 0.5,
    colors: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Overlay segmentation mask on image
    
    Args:
        image (np.ndarray): Original image [H, W, 3]
        mask (np.ndarray): Segmentation mask [H, W]
        alpha (float): Transparency of overlay
        colors (List[Tuple]): RGB colors for each class
    
    Returns:
        np.ndarray: Image with overlaid mask
    """
    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = create_color_mask(mask, colors=colors)
    
    # Overlay mask on image
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay


def plot_sample_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot a batch of images with masks and predictions
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        masks (torch.Tensor): Batch of ground truth masks [B, H, W]
        predictions (torch.Tensor, optional): Batch of predictions [B, C, H, W] or [B, H, W]
        num_samples (int): Number of samples to plot
        figsize (tuple): Figure size
        save_path (str, optional): Path to save plot
    """
    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Determine number of columns
    num_cols = 3 if predictions is not None else 2
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        image = denormalize_image(images[i])
        mask = masks[i].cpu().numpy()
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        colored_mask = create_color_mask(mask)
        overlay_gt = overlay_mask_on_image(image, mask, alpha=0.4)
        axes[i, 1].imshow(overlay_gt)
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Plot predictions if available
        if predictions is not None:
            if predictions.dim() == 4:  # [B, C, H, W]
                pred_mask = torch.argmax(predictions[i], dim=0).cpu().numpy()
            else:  # [B, H, W]
                pred_mask = predictions[i].cpu().numpy()
            
            overlay_pred = overlay_mask_on_image(image, pred_mask, alpha=0.4)
            axes[i, 2].imshow(overlay_pred)
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training history
    
    Args:
        history (dict): Training history with metrics
        save_path (str, optional): Path to save plot
        figsize (tuple): Figure size
    """
    metrics = list(history.keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        values = history[metric]
        epochs = range(1, len(values) + 1)
        
        axes[i].plot(epochs, values, 'b-', linewidth=2)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_class_distribution(
    dataset_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot class distribution in dataset
    
    Args:
        dataset_path (str): Path to dataset
        save_path (str, optional): Path to save plot
        figsize (tuple): Figure size
    """
    from src.data.dataset import PlantDiseaseDataset
    
    # Load dataset
    dataset = PlantDiseaseDataset(dataset_path, split='train')
    
    # Count pixels for each class
    class_counts = {0: 0, 1: 0}  # healthy, diseased
    
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample['mask'].numpy()
        
        for class_id in range(2):
            class_counts[class_id] += (mask == class_id).sum()
    
    # Plot distribution
    classes = ['Healthy', 'Diseased']
    counts = [class_counts[0], class_counts[1]]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(classes, counts, color=['green', 'red'], alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Pixels')
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def create_segmentation_report(
    model,
    test_loader,
    device: str = 'cuda',
    save_dir: str = 'results',
    num_samples: int = 10
):
    """
    Create comprehensive segmentation report
    
    Args:
        model: Trained segmentation model
        test_loader: Test data loader
        device (str): Device to run inference on
        save_dir (str): Directory to save results
        num_samples (int): Number of samples to visualize
    """
    from src.utils.metrics import SegmentationMetrics
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    metrics = SegmentationMetrics(num_classes=2)
    
    all_images = []
    all_masks = []
    all_predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(predictions, masks)
            
            # Store samples for visualization
            if len(all_images) < num_samples:
                all_images.extend(images.cpu())
                all_masks.extend(masks.cpu())
                all_predictions.extend(predictions.cpu())
            
            if i >= 10:  # Limit evaluation for demo
                break
    
    # Compute final metrics
    final_metrics = metrics.compute_all_metrics()
    
    # Save metrics
    metrics_text = "Segmentation Evaluation Results\n" + "="*40 + "\n"
    for key, value in final_metrics.items():
        metrics_text += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
    
    with open(save_dir / 'metrics.txt', 'w') as f:
        f.write(metrics_text)
    
    # Create visualizations
    plot_sample_batch(
        torch.stack(all_images[:num_samples]),
        torch.stack(all_masks[:num_samples]),
        torch.stack(all_predictions[:num_samples]),
        num_samples=min(num_samples, len(all_images)),
        save_path=str(save_dir / 'sample_results.png')
    )
    
    print(f"Segmentation report saved to {save_dir}")
    print("\nFinal Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")


def visualize_model_predictions(
    model,
    image_path: str,
    device: str = 'cuda',
    image_size: Tuple[int, int] = (512, 512),
    save_path: Optional[str] = None
):
    """
    Visualize model predictions on a single image
    
    Args:
        model: Trained segmentation model
        image_path (str): Path to input image
        device (str): Device to run inference on
        image_size (tuple): Input image size
        save_path (str, optional): Path to save visualization
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    # Preprocessing transforms
    transforms = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transforms(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize prediction back to original size
    prediction_resized = cv2.resize(
        prediction.astype(np.uint8), 
        (original_image.shape[1], original_image.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction mask
    colored_mask = create_color_mask(prediction_resized)
    axes[1].imshow(colored_mask)
    axes[1].set_title('Disease Segmentation')
    axes[1].axis('off')
    
    # Overlay
    overlay = overlay_mask_on_image(original_image, prediction_resized, alpha=0.4)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Calculate disease percentage
    total_pixels = prediction_resized.size
    diseased_pixels = (prediction_resized == 1).sum()
    disease_percentage = (diseased_pixels / total_pixels) * 100
    
    print(f"Disease coverage: {disease_percentage:.2f}%")
    
    return prediction_resized, disease_percentage


if __name__ == "__main__":
    # Test visualization functions
    batch_size, channels, height, width = 4, 3, 256, 256
    
    # Create dummy data
    images = torch.randn(batch_size, channels, height, width)
    masks = torch.randint(0, 2, (batch_size, height, width))
    predictions = torch.randn(batch_size, 2, height, width)
    
    # Test plotting
    plot_sample_batch(images, masks, predictions, num_samples=2)
    
    # Test training history plotting
    history = {
        'loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'iou': [0.5, 0.6, 0.7, 0.8, 0.85],
        'dice': [0.6, 0.7, 0.75, 0.8, 0.85]
    }
    plot_training_history(history)


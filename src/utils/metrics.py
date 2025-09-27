"""
Evaluation metrics for plant disease segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """
    Class to compute various segmentation metrics
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_intersection = torch.zeros(self.num_classes)
        self.total_union = torch.zeros(self.num_classes)
        self.total_target = torch.zeros(self.num_classes)
        self.total_prediction = torch.zeros(self.num_classes)
        self.total_correct = 0
        self.total_pixels = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions and targets
        
        Args:
            predictions (torch.Tensor): Model predictions [B, H, W] or [B, C, H, W]
            targets (torch.Tensor): Ground truth masks [B, H, W]
        """
        # Handle different input formats
        if predictions.dim() == 4:  # [B, C, H, W]
            predictions = torch.argmax(predictions, dim=1)  # [B, H, W]
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignored pixels
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Compute intersection and union for each class
        for class_id in range(self.num_classes):
            pred_class = (predictions == class_id)
            target_class = (targets == class_id)
            
            intersection = (pred_class & target_class).sum().float()
            union = (pred_class | target_class).sum().float()
            
            self.total_intersection[class_id] += intersection
            self.total_union[class_id] += union
            self.total_target[class_id] += target_class.sum().float()
            self.total_prediction[class_id] += pred_class.sum().float()
        
        # Update pixel accuracy
        self.total_correct += (predictions == targets).sum().float()
        self.total_pixels += targets.numel()
    
    def compute_iou(self) -> Dict[str, float]:
        """Compute Intersection over Union (IoU) for each class"""
        iou_per_class = self.total_intersection / (self.total_union + 1e-8)
        
        results = {}
        for class_id in range(self.num_classes):
            class_name = f"class_{class_id}"
            results[f"iou_{class_name}"] = iou_per_class[class_id].item()
        
        results["mean_iou"] = iou_per_class.mean().item()
        return results
    
    def compute_dice(self) -> Dict[str, float]:
        """Compute Dice coefficient for each class"""
        dice_per_class = (2 * self.total_intersection) / (
            self.total_target + self.total_prediction + 1e-8
        )
        
        results = {}
        for class_id in range(self.num_classes):
            class_name = f"class_{class_id}"
            results[f"dice_{class_name}"] = dice_per_class[class_id].item()
        
        results["mean_dice"] = dice_per_class.mean().item()
        return results
    
    def compute_pixel_accuracy(self) -> float:
        """Compute pixel-wise accuracy"""
        return (self.total_correct / (self.total_pixels + 1e-8)).item()
    
    def compute_precision_recall(self) -> Dict[str, float]:
        """Compute precision and recall for each class"""
        precision = self.total_intersection / (self.total_prediction + 1e-8)
        recall = self.total_intersection / (self.total_target + 1e-8)
        
        results = {}
        for class_id in range(self.num_classes):
            class_name = f"class_{class_id}"
            results[f"precision_{class_name}"] = precision[class_id].item()
            results[f"recall_{class_name}"] = recall[class_id].item()
        
        results["mean_precision"] = precision.mean().item()
        results["mean_recall"] = recall.mean().item()
        
        return results
    
    def compute_f1_score(self) -> Dict[str, float]:
        """Compute F1 score for each class"""
        precision = self.total_intersection / (self.total_prediction + 1e-8)
        recall = self.total_intersection / (self.total_target + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        results = {}
        for class_id in range(self.num_classes):
            class_name = f"class_{class_id}"
            results[f"f1_{class_name}"] = f1[class_id].item()
        
        results["mean_f1"] = f1.mean().item()
        return results
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {}
        metrics.update(self.compute_iou())
        metrics.update(self.compute_dice())
        metrics.update(self.compute_precision_recall())
        metrics.update(self.compute_f1_score())
        metrics["pixel_accuracy"] = self.compute_pixel_accuracy()
        
        return metrics


def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-8) -> torch.Tensor:
    """
    Compute Dice loss for segmentation
    
    Args:
        predictions (torch.Tensor): Model predictions [B, C, H, W]
        targets (torch.Tensor): Ground truth masks [B, H, W]
        smooth (float): Smoothing factor
    
    Returns:
        torch.Tensor: Dice loss
    """
    # Apply softmax to predictions
    predictions = F.softmax(predictions, dim=1)
    
    # Convert targets to one-hot encoding
    num_classes = predictions.shape[1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Flatten tensors
    predictions = predictions.view(predictions.shape[0], predictions.shape[1], -1)
    targets_one_hot = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
    
    # Compute Dice coefficient
    intersection = (predictions * targets_one_hot).sum(dim=2)
    dice = (2 * intersection + smooth) / (
        predictions.sum(dim=2) + targets_one_hot.sum(dim=2) + smooth
    )
    
    # Return Dice loss (1 - Dice coefficient)
    return 1 - dice.mean()


def focal_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    alpha: float = 1.0, 
    gamma: float = 2.0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute Focal loss for addressing class imbalance
    
    Args:
        predictions (torch.Tensor): Model predictions [B, C, H, W]
        targets (torch.Tensor): Ground truth masks [B, H, W]
        alpha (float): Weighting factor for rare class
        gamma (float): Focusing parameter
        reduction (str): Reduction method
    
    Returns:
        torch.Tensor: Focal loss
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def combined_loss(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    ce_weight: float = 0.5, 
    dice_weight: float = 0.3,
    focal_weight: float = 0.2
) -> torch.Tensor:
    """
    Combine multiple loss functions for better segmentation
    
    Args:
        predictions (torch.Tensor): Model predictions [B, C, H, W]
        targets (torch.Tensor): Ground truth masks [B, H, W]
        ce_weight (float): Weight for cross-entropy loss
        dice_weight (float): Weight for Dice loss
        focal_weight (float): Weight for Focal loss
    
    Returns:
        torch.Tensor: Combined loss
    """
    ce_loss = F.cross_entropy(predictions, targets)
    dice_loss_val = dice_loss(predictions, targets)
    focal_loss_val = focal_loss(predictions, targets)
    
    total_loss = (
        ce_weight * ce_loss + 
        dice_weight * dice_loss_val + 
        focal_weight * focal_loss_val
    )
    
    return total_loss


def compute_confusion_matrix(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    num_classes: int = 2
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth
        num_classes (int): Number of classes
    
    Returns:
        np.ndarray: Confusion matrix
    """
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    predictions = predictions.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: List[str], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (List[str]): Class names
        save_path (str, optional): Path to save plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_class_weights(targets: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        targets (torch.Tensor): Ground truth masks
        num_classes (int): Number of classes
    
    Returns:
        torch.Tensor: Class weights
    """
    # Count pixels for each class
    class_counts = torch.zeros(num_classes)
    
    for class_id in range(num_classes):
        class_counts[class_id] = (targets == class_id).sum().float()
    
    # Calculate weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights


if __name__ == "__main__":
    # Test metrics
    batch_size, height, width = 4, 256, 256
    num_classes = 2
    
    # Create dummy data
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test metrics
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(predictions, targets)
    
    results = metrics.compute_all_metrics()
    print("Metrics:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Test losses
    ce_loss = F.cross_entropy(predictions, targets)
    dice_loss_val = dice_loss(predictions, targets)
    focal_loss_val = focal_loss(predictions, targets)
    combined_loss_val = combined_loss(predictions, targets)
    
    print(f"\nLosses:")
    print(f"Cross Entropy: {ce_loss:.4f}")
    print(f"Dice Loss: {dice_loss_val:.4f}")
    print(f"Focal Loss: {focal_loss_val:.4f}")
    print(f"Combined Loss: {combined_loss_val:.4f}")
    
    # Test confusion matrix
    cm = compute_confusion_matrix(predictions, targets, num_classes)
    print(f"\nConfusion Matrix:\n{cm}")


"""
Training script for plant disease segmentation using ENet
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from src.models.enet import create_enet_model
from src.data.dataset import PlantDiseaseDataModule
from src.utils.metrics import SegmentationMetrics, combined_loss, dice_loss, focal_loss
from src.utils.visualization import plot_sample_batch, plot_training_history
from src.utils.config import Config


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.monitor_op = np.greater if mode == 'max' else np.less
        self.min_delta *= 1 if mode == 'max' else -1
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class PlantDiseaseTrainer:
    """Trainer class for plant disease segmentation"""
    
    def __init__(self, config: Config, experiment_dir: Path):
        self.config = config
        self.experiment_dir = experiment_dir
        
        # Setup device
        device_str = config.get('hardware.device', 'cpu')
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        
        # Initialize model
        self.model = create_enet_model(
            num_classes=config.get('model.num_classes', 2)
        ).to(self.device)
        
        # Setup data
        self.data_module = PlantDiseaseDataModule(
            data_dir=config.get('data.data_dir'),
            batch_size=config.get('data.batch_size', 16),
            image_size=config.get('data.image_size', [512, 512]),
            num_workers=config.get('data.num_workers', 4),
            normalize=config.get('data.normalize', True),
            pin_memory=config.get('data.pin_memory', True)
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup metrics
        self.train_metrics = SegmentationMetrics(config.get('model.num_classes', 2))
        self.val_metrics = SegmentationMetrics(config.get('model.num_classes', 2))
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get('training.early_stopping.patience', 15),
            min_delta=config.get('training.early_stopping.min_delta', 0.001),
            mode=config.get('training.early_stopping.mode', 'max')
        )
        
        # Setup logging
        self.writer = None
        if config.get('logging.tensorboard', True):
            log_dir = experiment_dir / 'logs'
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': []
        }
        
        # Mixed precision training
        self.use_amp = config.get('training.use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        print(f"Trainer initialized with device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_name = self.config.get('training.optimizer', 'adam').lower()
        lr = float(self.config.get('training.learning_rate', 0.001))
        weight_decay = float(self.config.get('training.weight_decay', 1e-4))
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('training.scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(self.config.get('training.epochs', 100))
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=10, factor=0.5
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration"""
        loss_type = self.config.get('training.loss.type', 'combined')
        
        if loss_type == 'ce':
            return nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            return dice_loss
        elif loss_type == 'focal':
            return lambda pred, target: focal_loss(
                pred, target,
                alpha=self.config.get('training.loss.focal_params.alpha', 1.0),
                gamma=self.config.get('training.loss.focal_params.gamma', 2.0)
            )
        elif loss_type == 'combined':
            return lambda pred, target: combined_loss(
                pred, target,
                ce_weight=self.config.get('training.loss.weights.ce', 0.5),
                dice_weight=self.config.get('training.loss.weights.dice', 0.3),
                focal_weight=self.config.get('training.loss.weights.focal', 0.2)
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} - Training')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('training.grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('training.grad_clip')
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.get('training.grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('training.grad_clip')
                    )
                
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=1)
                self.train_metrics.update(predictions, masks)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute_all_metrics()
        
        return {'loss': avg_loss, **metrics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} - Validation')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Update metrics
                predictions = torch.argmax(outputs, dim=1)
                self.val_metrics.update(predictions, masks)
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute_all_metrics()
        
        return {'loss': avg_loss, **metrics}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = self.experiment_dir / 'checkpoints'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
        
        # Save periodic checkpoint
        if epoch % self.config.get('logging.save_frequency', 10) == 0:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('scaler_state_dict') and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard and console"""
        if self.writer:
            # Log scalars
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Log learning rate
            if self.scheduler:
                self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
        
        # Console logging
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['mean_iou']:.4f}, Dice: {train_metrics['mean_dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['mean_iou']:.4f}, Dice: {val_metrics['mean_dice']:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Setup data loaders
        self.data_module.setup('fit')
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        resume_path = self.config.get('checkpoint.resume_from')
        if resume_path and Path(resume_path).exists():
            start_epoch = self.load_checkpoint(resume_path) + 1
        
        # Training loop
        for epoch in range(start_epoch, self.config.get('training.epochs', 100)):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            if epoch % self.config.get('validation.frequency', 1) == 0:
                val_metrics = self.validate_epoch(val_loader)
            else:
                val_metrics = {'loss': 0.0, 'mean_iou': 0.0, 'mean_dice': 0.0}
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_iou'].append(train_metrics['mean_iou'])
            self.training_history['val_iou'].append(val_metrics['mean_iou'])
            self.training_history['train_dice'].append(train_metrics['mean_dice'])
            self.training_history['val_dice'].append(val_metrics['mean_dice'])
            
            # Check for best model
            current_metric = val_metrics[self.config.get('validation.metric', 'mean_iou')]
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping(current_metric):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print("-" * 80)
        
        # Save final training history
        plot_training_history(
            self.training_history,
            save_path=str(self.experiment_dir / 'results' / 'training_history.png')
        )
        
        if self.writer:
            self.writer.close()
        
        print("Training completed!")
        print(f"Best validation metric: {self.best_metric:.4f}")


if __name__ == "__main__":
    from src.utils.config import parse_args, override_config_from_args, setup_seed, create_experiment_dir, validate_config
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    override_config_from_args(config, args)
    validate_config(config)
    
    # Setup reproducibility
    setup_seed(config)
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(config, args.experiment_name)
    
    # Initialize trainer
    trainer = PlantDiseaseTrainer(config, experiment_dir)
    
    # Start training
    trainer.train()


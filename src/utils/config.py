"""
Configuration management utilities
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch


class Config:
    """Configuration class for managing experiment settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path (str, optional): Path to YAML configuration file
        """
        self.config = {}
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key (str): Configuration key (e.g., 'model.num_classes')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key (str): Configuration key (e.g., 'model.num_classes')
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self.config
        
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]
        
        config_dict[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]):
        """Update configuration with another dictionary"""
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration"""
        return self.get(key) is not None


def setup_device(config: Config) -> torch.device:
    """
    Setup computing device based on configuration
    
    Args:
        config (Config): Configuration object
    
    Returns:
        torch.device: Device to use for computation
    """
    device_config = config.get('hardware.device', 'auto')
    
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU instead")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    elif device_config == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')  # Default to CPU for safety
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def setup_seed(config: Config):
    """Setup random seed for reproducibility"""
    seed = config.get('system.seed', 42)
    
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if config.get('system.deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif config.get('system.benchmark', True):
        torch.backends.cudnn.benchmark = True
    
    print(f"Random seed set to: {seed}")


def create_experiment_dir(config: Config, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        config (Config): Configuration object
        experiment_name (str): Name of the experiment
    
    Returns:
        Path: Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{experiment_name}_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration to experiment directory
    config.save_to_file(exp_dir / "config.yaml")
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    print(f"Experiment directory created: {exp_dir}")
    
    return exp_dir


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Plant Disease Segmentation")
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to data directory (overrides config)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--learning_rate', '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='plant_disease_segmentation',
        help='Name of the experiment'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (smaller dataset, fewer epochs)'
    )
    
    return parser.parse_args()


def override_config_from_args(config: Config, args: argparse.Namespace):
    """Override configuration values from command line arguments"""
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
        # Debug mode: smaller dataset, fewer epochs
        config.set('training.epochs', 5)
        config.set('data.batch_size', 4)
        config.set('validation.frequency', 1)
        print("Debug mode enabled: reduced epochs and batch size")


def validate_config(config: Config):
    """Validate configuration values"""
    required_keys = [
        'model.num_classes',
        'data.data_dir',
        'data.batch_size',
        'training.epochs',
        'training.learning_rate'
    ]
    
    for key in required_keys:
        if config.get(key) is None:
            raise ValueError(f"Required configuration key missing: {key}")
    
    # Validate numeric values
    num_classes = config.get('model.num_classes')
    if isinstance(num_classes, (int, float)) and num_classes <= 0:
        raise ValueError("Number of classes must be positive")
    
    batch_size = config.get('data.batch_size')
    if isinstance(batch_size, (int, float)) and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    epochs = config.get('training.epochs')
    if isinstance(epochs, (int, float)) and epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    learning_rate = config.get('training.learning_rate')
    if isinstance(learning_rate, (int, float)) and learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    # Validate paths
    data_dir = Path(config.get('data.data_dir'))
    if not data_dir.exists():
        print(f"Warning: Data directory does not exist: {data_dir}")
    
    print("Configuration validation passed")


if __name__ == "__main__":
    # Test configuration loading
    config = Config('configs/config.yaml')
    
    print("Configuration loaded:")
    print(f"Model: {config.get('model.name')}")
    print(f"Classes: {config.get('model.num_classes')}")
    print(f"Batch size: {config.get('data.batch_size')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    
    # Test device setup
    device = setup_device(config)
    
    # Test seed setup
    setup_seed(config)


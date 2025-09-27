# Plant Disease Segmentation System

A comprehensive deep learning system for segmenting plant diseases using ENet architecture. This system is designed for precision pesticide spraying applications, enabling targeted treatment of diseased plant areas while minimizing pesticide wastage.

## 🌟 Features

- **ENet Architecture**: Efficient semantic segmentation model optimized for real-time inference
- **Comprehensive Training Pipeline**: Complete training system with data augmentation, multiple loss functions, and early stopping
- **Advanced Metrics**: IoU, Dice coefficient, precision, recall, F1-score, and pixel accuracy
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Visualization Tools**: Rich visualization capabilities for training monitoring and result analysis
- **Batch Processing**: Efficient inference on single images or entire directories
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Tensorboard Integration**: Real-time training monitoring

## 📁 Project Structure

```
krishiRakshak/
├── src/
│   ├── models/
│   │   └── enet.py              # ENet model implementation
│   ├── data/
│   │   └── dataset.py           # Dataset loading and preprocessing
│   ├── training/
│   │   └── trainer.py           # Training pipeline
│   ├── inference/
│   │   └── predictor.py         # Inference pipeline
│   └── utils/
│       ├── config.py            # Configuration management
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Visualization utilities
├── configs/
│   └── config.yaml              # Default configuration
├── data/                        # Data directory
│   ├── raw/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── masks/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── processed/
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── results/                     # Results and visualizations
├── main.py                      # Main pipeline script
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd krishiRakshak

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your data in the following structure:

```
data/raw/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
└── masks/
    ├── train/          # Training masks (binary: 0=healthy, 1=diseased)
    ├── val/            # Validation masks
    └── test/           # Test masks
```

Or use the data preparation utility:

```bash
python main.py prepare_data --source_data_dir /path/to/raw/data --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1
```

### 3. Training

```bash
# Train with default configuration
python main.py train

# Train with custom configuration
python main.py train --config configs/custom_config.yaml --epochs 50 --batch_size 8

# Resume training from checkpoint
python main.py train --resume checkpoints/latest.pth
```

### 4. Inference

```bash
# Single image inference
python main.py inference --model_path checkpoints/best.pth --input_path image.jpg --output_path results/

# Batch inference on directory
python main.py inference --model_path checkpoints/best.pth --input_path images_dir/ --output_path results/
```

### 5. Evaluation

```bash
# Evaluate model on test set
python main.py evaluate --model_path checkpoints/best.pth --output_path evaluation_results/
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization. Key configuration sections:

### Model Configuration
```yaml
model:
  name: "enet"
  num_classes: 2
  encoder_relu: false
  decoder_relu: true
```

### Training Configuration
```yaml
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  loss:
    type: "combined"
    weights:
      ce: 0.5
      dice: 0.3
      focal: 0.2
```

### Data Configuration
```yaml
data:
  data_dir: "data"
  image_size: [512, 512]
  batch_size: 16
  normalize: true
```

## 📊 Metrics and Evaluation

The system provides comprehensive evaluation metrics:

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Precision & Recall**: Per-class and mean values
- **F1-Score**: Harmonic mean of precision and recall

## 🎯 Loss Functions

Multiple loss functions are supported:

1. **Cross-Entropy Loss**: Standard classification loss
2. **Dice Loss**: Optimizes for overlap between prediction and ground truth
3. **Focal Loss**: Addresses class imbalance by focusing on hard examples
4. **Combined Loss**: Weighted combination of multiple losses

## 📈 Monitoring and Visualization

- **Tensorboard Integration**: Real-time training monitoring
- **Training History Plots**: Loss and metrics over epochs
- **Segmentation Visualizations**: Original image, prediction, and overlay
- **Confusion Matrix**: Classification performance analysis
- **Class Distribution**: Dataset statistics

## 🔧 Advanced Features

### Mixed Precision Training
Reduces memory usage and speeds up training:
```yaml
training:
  use_amp: true
```

### Early Stopping
Prevents overfitting:
```yaml
training:
  early_stopping:
    patience: 15
    min_delta: 0.001
    monitor: "val_iou"
```

### Data Augmentation
Comprehensive augmentation pipeline:
```yaml
data:
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.3
    rotate90: 0.5
    brightness_contrast:
      brightness_limit: 0.2
      contrast_limit: 0.2
```

## 🚀 Usage Examples

### Training from Scratch
```bash
python main.py train \
    --data_dir data/ \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --experiment_name my_experiment
```

### Inference on Test Images
```bash
python main.py inference \
    --model_path experiments/my_experiment_20231201_120000/checkpoints/best.pth \
    --input_path test_images/ \
    --output_path inference_results/
```

### Custom Configuration
```bash
python main.py train \
    --config configs/high_resolution.yaml \
    --device cuda \
    --experiment_name high_res_experiment
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV 4.8+
- Albumentations 1.3+
- NumPy 1.24+
- Matplotlib 3.7+
- Tensorboard 2.13+

## 🎯 Applications

This system is specifically designed for:

1. **Precision Agriculture**: Targeted pesticide application
2. **Crop Monitoring**: Early disease detection
3. **Yield Optimization**: Minimizing crop loss through timely intervention
4. **Environmental Conservation**: Reducing chemical usage through precision targeting
5. **Cost Reduction**: Optimizing pesticide usage and application costs

## 📝 Model Architecture

The ENet architecture consists of:

- **Initial Block**: Efficient downsampling
- **Bottleneck Blocks**: Regular, downsampling, and upsampling variants
- **Dilated Convolutions**: Increased receptive field
- **Asymmetric Convolutions**: Computational efficiency
- **Skip Connections**: Feature preservation

## 🔍 Performance Optimization

- **Mixed Precision Training**: Faster training with reduced memory
- **Gradient Clipping**: Stable training
- **Cosine Annealing**: Optimal learning rate scheduling
- **Data Loading Optimization**: Multi-worker data loading with pinned memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ENet architecture based on the original paper by Paszke et al.
- PyTorch community for the excellent deep learning framework
- Albumentations for comprehensive data augmentation

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Provide code examples and error messages when applicable

---

**Note**: This system is designed for research and educational purposes. For production deployment, additional testing and validation are recommended.



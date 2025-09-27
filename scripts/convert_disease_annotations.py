"""
Convert JSON annotations to proper disease segmentation masks
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import shutil


def create_combined_dataset(
    healthy_dir,
    diseased_dir,
    output_dir,
    visualize=True
):
    """
    Create combined dataset with proper multi-class segmentation masks
    
    Args:
        healthy_dir: Directory with healthy leaf annotations
        diseased_dir: Directory with diseased leaf annotations  
        output_dir: Output directory for combined dataset
        visualize: Whether to create sample visualizations
    """
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating combined disease segmentation dataset...")
    print("Classes: 0=Background, 1=Healthy_Leaf, 2=Diseased_Area")
    
    processed_count = 0
    
    # Process healthy dataset
    print("\nProcessing healthy leaf dataset...")
    healthy_path = Path(healthy_dir)
    healthy_json_files = list(healthy_path.glob('*.json'))
    
    for json_file in tqdm(healthy_json_files, desc="Processing healthy leaves"):
        try:
            processed_count += process_annotation_file(
                json_file, healthy_path, images_dir, masks_dir, 
                label_mapping={'Healthy': 1}, 
                file_prefix='healthy'
            )
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Process diseased dataset  
    print("\nProcessing diseased leaf dataset...")
    diseased_path = Path(diseased_dir)
    diseased_json_files = list(diseased_path.glob('*.json'))
    
    for json_file in tqdm(diseased_json_files, desc="Processing diseased leaves"):
        try:
            processed_count += process_annotation_file(
                json_file, diseased_path, images_dir, masks_dir,
                label_mapping={'bacteria': 2, 'Bacteria': 2, 'diseased': 2, 'Diseased': 2},
                file_prefix='diseased'
            )
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"\nDataset creation completed!")
    print(f"Total processed images: {processed_count}")
    print(f"Output directory: {output_path}")
    
    # Create statistics and visualization
    create_dataset_statistics(output_path)
    
    if visualize:
        create_sample_visualization(output_path)


def process_annotation_file(json_file, source_dir, images_dir, masks_dir, label_mapping, file_prefix):
    """Process a single annotation file"""
    # Load annotation
    with open(json_file, 'r') as f:
        annotation = json.load(f)
    
    # Find corresponding image
    image_filename = annotation.get('imagePath', json_file.stem + '.JPG')
    image_path = source_dir / image_filename
    
    if not image_path.exists():
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = source_dir / (json_file.stem + ext)
            if test_path.exists():
                image_path = test_path
                break
        
        if not image_path.exists():
            print(f"Warning: Image not found for {json_file.name}")
            return 0
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return 0
    
    height, width = image.shape[:2]
    
    # Create mask
    mask = create_segmentation_mask((height, width), annotation['shapes'], label_mapping)
    
    # Generate unique filename
    base_name = f"{file_prefix}_{json_file.stem}"
    
    # Save image
    output_image_path = images_dir / f"{base_name}.jpg"
    cv2.imwrite(str(output_image_path), image)
    
    # Save mask
    output_mask_path = masks_dir / f"{base_name}.png"
    cv2.imwrite(str(output_mask_path), mask)
    
    return 1


def create_segmentation_mask(image_shape, shapes, label_mapping):
    """Create segmentation mask from polygon annotations"""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for shape in shapes:
        label = shape['label']
        if label not in label_mapping:
            continue
            
        class_id = label_mapping[label]
        points = shape['points']
        
        if shape.get('shape_type', 'polygon') == 'polygon':
            # Convert points to integer coordinates
            polygon_points = [(int(x), int(y)) for x, y in points]
            
            # Create PIL image for drawing
            pil_mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(pil_mask)
            
            # Draw filled polygon
            draw.polygon(polygon_points, fill=class_id)
            
            # Convert back to numpy array
            polygon_mask = np.array(pil_mask)
            
            # Update mask (use maximum to handle overlapping polygons)
            mask = np.maximum(mask, polygon_mask)
    
    return mask


def create_dataset_statistics(dataset_path):
    """Create and display dataset statistics"""
    masks_dir = dataset_path / 'masks'
    
    if not masks_dir.exists():
        return
    
    print("\nDataset Statistics:")
    print("-" * 40)
    
    mask_files = list(masks_dir.glob('*.png'))
    total_pixels = 0
    class_pixels = {0: 0, 1: 0, 2: 0}  # Background, Healthy, Diseased
    
    for mask_file in tqdm(mask_files, desc="Computing statistics"):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique, counts = np.unique(mask, return_counts=True)
            total_pixels += mask.size
            
            for class_id, count in zip(unique, counts):
                if class_id in class_pixels:
                    class_pixels[class_id] += count
    
    print(f"Total images: {len(mask_files)}")
    print(f"Total pixels: {total_pixels:,}")
    
    class_names = {0: 'Background', 1: 'Healthy_Leaf', 2: 'Diseased_Area'}
    
    for class_id, pixel_count in class_pixels.items():
        class_name = class_names[class_id]
        percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"{class_name}: {pixel_count:,} pixels ({percentage:.2f}%)")


def create_sample_visualization(dataset_path, num_samples=5):
    """Create sample visualization of the dataset"""
    import matplotlib.pyplot as plt
    
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    # Get sample files
    image_files = list(images_dir.glob('*.jpg'))[:num_samples]
    
    if not image_files:
        print("No images found for visualization")
        return
    
    fig, axes = plt.subplots(3, len(image_files), figsize=(15, 9))
    if len(image_files) == 1:
        axes = axes.reshape(-1, 1)
    
    colors = np.array([[0, 0, 0],      # Background - Black
                      [0, 255, 0],     # Healthy - Green  
                      [255, 0, 0]])    # Diseased - Red
    
    for i, image_file in enumerate(image_files):
        # Load image
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        mask_file = masks_dir / f"{image_file.stem}.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Create colored mask
        colored_mask = colors[mask]
        
        # Create overlay
        overlay = cv2.addWeighted(image, 0.7, colored_mask.astype(np.uint8), 0.3, 0)
        
        # Plot original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask, cmap='viridis', vmin=0, vmax=2)
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
        
        # Plot overlay
        axes[2, i].imshow(overlay)
        axes[2, i].set_title(f'Overlay {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(dataset_path / 'dataset_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset visualization saved to {dataset_path / 'dataset_visualization.png'}")


def create_train_val_test_splits(dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Create train/validation/test splits"""
    from sklearn.model_selection import train_test_split
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    # Get all image files
    image_files = list(images_dir.glob('*.jpg'))
    image_names = [f.stem for f in image_files]
    
    print(f"\nCreating train/val/test splits...")
    print(f"Total images: {len(image_names)}")
    
    # Create splits
    train_names, temp_names = train_test_split(
        image_names, test_size=(1 - train_ratio), random_state=42, shuffle=True
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_names, test_names = train_test_split(
        temp_names, test_size=(1 - val_size), random_state=42, shuffle=True
    )
    
    print(f"Train: {len(train_names)} images")
    print(f"Validation: {len(val_names)} images") 
    print(f"Test: {len(test_names)} images")
    
    # Create split directories
    splits = {'train': train_names, 'val': val_names, 'test': test_names}
    
    for split_name, file_names in splits.items():
        # Create directories
        split_images_dir = dataset_path / 'raw' / 'images' / split_name
        split_masks_dir = dataset_path / 'raw' / 'masks' / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for file_name in file_names:
            # Copy image
            src_image = images_dir / f"{file_name}.jpg"
            dst_image = split_images_dir / f"{file_name}.jpg"
            shutil.copy2(src_image, dst_image)
            
            # Copy mask
            src_mask = masks_dir / f"{file_name}.png"
            dst_mask = split_masks_dir / f"{file_name}.png"
            shutil.copy2(src_mask, dst_mask)
    
    print(f"Data splits created in {dataset_path / 'raw'}")
    return dataset_path / 'raw'


def main():
    parser = argparse.ArgumentParser(description="Create disease segmentation dataset")
    parser.add_argument('--healthy_dir', type=str, required=True,
                       help='Directory with healthy leaf annotations')
    parser.add_argument('--diseased_dir', type=str, required=True,
                       help='Directory with diseased leaf annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for combined dataset')
    parser.add_argument('--create_splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    # Create combined dataset
    create_combined_dataset(
        healthy_dir=args.healthy_dir,
        diseased_dir=args.diseased_dir,
        output_dir=args.output_dir,
        visualize=True
    )
    
    # Create splits if requested
    if args.create_splits:
        create_train_val_test_splits(
            dataset_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )


if __name__ == "__main__":
    main()

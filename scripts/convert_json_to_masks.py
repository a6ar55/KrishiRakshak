"""
Convert JSON annotations (LabelMe format) to segmentation masks
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


def load_json_annotation(json_path):
    """Load JSON annotation file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_mask_from_polygons(image_shape, shapes, label_mapping):
    """
    Create segmentation mask from polygon annotations
    
    Args:
        image_shape: (height, width) of the image
        shapes: List of shape annotations from JSON
        label_mapping: Dictionary mapping label names to class IDs
    
    Returns:
        np.ndarray: Segmentation mask
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for shape in shapes:
        label = shape['label']
        if label not in label_mapping:
            print(f"Warning: Unknown label '{label}', skipping...")
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


def process_dataset(
    source_dir,
    output_dir,
    label_mapping=None,
    create_binary_masks=True
):
    """
    Process entire dataset of JSON annotations to create masks
    
    Args:
        source_dir: Directory containing images and JSON files
        output_dir: Output directory for processed data
        label_mapping: Dictionary mapping labels to class IDs
        create_binary_masks: If True, create binary masks (leaf vs background)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / 'images'
    masks_dir = output_path / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Default label mapping
    if label_mapping is None:
        if create_binary_masks:
            # Binary segmentation: background (0) vs leaf (1)
            label_mapping = {'Healthy': 1}
        else:
            # Multi-class: background (0), healthy (1), diseased (2)
            label_mapping = {'Healthy': 1, 'Diseased': 2}
    
    print(f"Label mapping: {label_mapping}")
    
    # Find all JSON files
    json_files = list(source_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON annotation files")
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in tqdm(json_files, desc="Processing annotations"):
        try:
            # Load annotation
            annotation = load_json_annotation(json_file)
            
            # Get corresponding image file
            image_filename = annotation.get('imagePath', json_file.stem + '.JPG')
            image_path = source_path / image_filename
            
            if not image_path.exists():
                # Try different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    test_path = source_path / (json_file.stem + ext)
                    if test_path.exists():
                        image_path = test_path
                        break
                
                if not image_path.exists():
                    print(f"Warning: Image not found for {json_file.name}")
                    skipped_count += 1
                    continue
            
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                skipped_count += 1
                continue
            
            height, width = image.shape[:2]
            
            # Create mask
            mask = create_mask_from_polygons(
                (height, width), 
                annotation['shapes'], 
                label_mapping
            )
            
            # Save image and mask
            base_name = json_file.stem
            
            # Copy image
            output_image_path = images_dir / f"{base_name}.jpg"
            shutil.copy2(image_path, output_image_path)
            
            # Save mask
            output_mask_path = masks_dir / f"{base_name}.png"
            cv2.imwrite(str(output_mask_path), mask)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output saved to: {output_path}")
    
    # Create statistics
    create_dataset_statistics(output_path, label_mapping)


def create_dataset_statistics(dataset_path, label_mapping):
    """Create dataset statistics"""
    masks_dir = dataset_path / 'masks'
    
    if not masks_dir.exists():
        return
    
    print("\nDataset Statistics:")
    print("-" * 30)
    
    mask_files = list(masks_dir.glob('*.png'))
    total_pixels = 0
    class_pixels = {class_id: 0 for class_id in label_mapping.values()}
    class_pixels[0] = 0  # Background
    
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
    
    # Reverse mapping for display
    id_to_label = {v: k for k, v in label_mapping.items()}
    id_to_label[0] = 'Background'
    
    for class_id, pixel_count in class_pixels.items():
        label = id_to_label.get(class_id, f'Class_{class_id}')
        percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"{label}: {pixel_count:,} pixels ({percentage:.2f}%)")


def visualize_samples(dataset_path, num_samples=5):
    """Visualize some sample image-mask pairs"""
    import matplotlib.pyplot as plt
    
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'
    
    image_files = list(images_dir.glob('*.jpg'))[:num_samples]
    
    if not image_files:
        print("No images found for visualization")
        return
    
    fig, axes = plt.subplots(2, len(image_files), figsize=(15, 6))
    if len(image_files) == 1:
        axes = axes.reshape(-1, 1)
    
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
        
        # Plot image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Plot mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(dataset_path / 'sample_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sample visualization saved to {dataset_path / 'sample_visualization.png'}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON annotations to segmentation masks")
    parser.add_argument('--source_dir', '-s', type=str, required=True,
                       help='Directory containing images and JSON annotations')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--binary_masks', action='store_true',
                       help='Create binary masks (leaf vs background)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of sample results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        create_binary_masks=args.binary_masks
    )
    
    # Create visualization if requested
    if args.visualize:
        visualize_samples(Path(args.output_dir), args.num_samples)


if __name__ == "__main__":
    main()



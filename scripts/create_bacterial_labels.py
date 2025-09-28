"""
Create label images from Bacterial Spot JSON annotations
Converts JSON files to PNG label images with "_label" suffix
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_json_annotation(json_path):
    """Load JSON annotation file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_label_mask(image_shape, shapes, class_mapping):
    """
    Create label mask from polygon annotations
    
    Args:
        image_shape: (height, width) of the image
        shapes: List of shape annotations from JSON
        class_mapping: Dictionary mapping label names to class IDs
    
    Returns:
        np.ndarray: Label mask with class IDs
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    print(f"Creating mask of size: {height} x {width}")
    
    for i, shape in enumerate(shapes):
        label = shape['label']
        points = shape['points']
        
        print(f"Processing shape {i+1}: label='{label}', points={len(points)}")
        
        if label not in class_mapping:
            print(f"Warning: Unknown label '{label}', skipping...")
            continue
            
        class_id = class_mapping[label]
        
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
            
            print(f"  - Drew polygon with {len(polygon_points)} points, class_id={class_id}")
    
    return mask


def process_bacterial_spot_dataset(
    source_dir,
    output_dir,
    visualize_samples=True
):
    """
    Process Bacterial Spot dataset to create label images
    
    Args:
        source_dir: Directory containing JSON annotations and images
        output_dir: Output directory for label images
        visualize_samples: Whether to create sample visualizations
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define class mapping for bacterial spot
    class_mapping = {
        'bacteria': 255,      # White pixels for diseased areas
        'Bacteria': 255,      # Handle capitalization variations
        'diseased': 255,
        'Diseased': 255
    }
    
    print(f"Processing Bacterial Spot dataset...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Class mapping: {class_mapping}")
    
    # Find all JSON files
    json_files = list(source_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON annotation files")
    
    if not json_files:
        print("No JSON files found! Please check the source directory.")
        return
    
    processed_count = 0
    skipped_count = 0
    sample_results = []
    
    for json_file in tqdm(json_files, desc="Processing annotations"):
        try:
            print(f"\nProcessing: {json_file.name}")
            
            # Load annotation
            annotation = load_json_annotation(json_file)
            
            # Get image dimensions from annotation or load image
            if 'imageHeight' in annotation and 'imageWidth' in annotation:
                height = annotation['imageHeight']
                width = annotation['imageWidth']
                print(f"Got dimensions from JSON: {width} x {height}")
            else:
                # Try to load corresponding image to get dimensions
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
                print(f"Got dimensions from image: {width} x {height}")
            
            # Create label mask
            label_mask = create_label_mask(
                (height, width), 
                annotation['shapes'], 
                class_mapping
            )
            
            # Create output filename with "_label" suffix
            base_name = json_file.stem  # e.g., "Bacterial Spot00001"
            output_filename = f"{base_name}_label.png"
            output_path_full = output_path / output_filename
            
            # Save label mask
            cv2.imwrite(str(output_path_full), label_mask)
            
            print(f"Saved: {output_filename}")
            print(f"Label statistics: min={label_mask.min()}, max={label_mask.max()}, unique={np.unique(label_mask)}")
            
            # Store sample for visualization
            if len(sample_results) < 5 and visualize_samples:
                # Try to load corresponding image for visualization
                image_filename = annotation.get('imagePath', json_file.stem + '.JPG')
                image_path = source_path / image_filename
                
                if not image_path.exists():
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_path = source_path / (json_file.stem + ext)
                        if test_path.exists():
                            image_path = test_path
                            break
                
                if image_path.exists():
                    original_image = cv2.imread(str(image_path))
                    if original_image is not None:
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                        sample_results.append({
                            'name': base_name,
                            'image': original_image,
                            'label': label_mask,
                            'output_file': output_filename
                        })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output directory: {output_path}")
    
    # Create visualization
    if visualize_samples and sample_results:
        create_visualization(sample_results, output_path)
    
    # Create summary statistics
    create_summary_statistics(output_path)


def create_visualization(sample_results, output_dir):
    """Create visualization of sample results"""
    print(f"\nCreating visualization with {len(sample_results)} samples...")
    
    fig, axes = plt.subplots(2, len(sample_results), figsize=(15, 8))
    if len(sample_results) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample in enumerate(sample_results):
        # Original image
        axes[0, i].imshow(sample['image'])
        axes[0, i].set_title(f"Original: {sample['name']}")
        axes[0, i].axis('off')
        
        # Label mask
        axes[1, i].imshow(sample['label'], cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f"Label: {sample['output_file']}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    viz_path = output_dir / 'bacterial_labels_visualization.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {viz_path}")


def create_summary_statistics(output_dir):
    """Create summary statistics of the generated labels"""
    print("\nGenerating summary statistics...")
    
    label_files = list(output_dir.glob('*_label.png'))
    
    if not label_files:
        print("No label files found for statistics.")
        return
    
    total_pixels = 0
    diseased_pixels = 0
    background_pixels = 0
    
    for label_file in tqdm(label_files, desc="Computing statistics"):
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        if label is not None:
            total_pixels += label.size
            diseased_pixels += (label == 255).sum()
            background_pixels += (label == 0).sum()
    
    # Write statistics to file
    stats_text = f"""Bacterial Spot Label Generation Summary
{'='*50}

Total label files created: {len(label_files)}
Total pixels: {total_pixels:,}
Background pixels (0): {background_pixels:,} ({background_pixels/total_pixels*100:.2f}%)
Diseased pixels (255): {diseased_pixels:,} ({diseased_pixels/total_pixels*100:.2f}%)

Label Format:
- Background/Healthy: 0 (black)
- Diseased/Bacteria: 255 (white)

Files created with "_label.png" suffix in: {output_dir}
"""
    
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write(stats_text)
    
    print(stats_text)


def main():
    parser = argparse.ArgumentParser(description="Create label images from Bacterial Spot JSON annotations")
    parser.add_argument('--source_dir', '-s', type=str, default='Bacterial Spot/Bacterial Spot',
                       help='Directory containing JSON annotations and images')
    parser.add_argument('--output_dir', '-o', type=str, default='Bacterial_labels',
                       help='Output directory for label images')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization creation')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_bacterial_spot_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        visualize_samples=not args.no_viz
    )


if __name__ == "__main__":
    main()

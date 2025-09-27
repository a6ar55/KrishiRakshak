"""
Script to download and prepare sample plant disease dataset
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil


def download_file(url: str, destination: Path):
    """Download file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {destination}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to {extract_to}")


def create_sample_dataset():
    """Create a sample dataset structure for testing"""
    print("Creating sample dataset structure...")
    
    # Create directory structure
    base_dir = Path("data/raw")
    
    for split in ['train', 'val', 'test']:
        (base_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base_dir / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    print("Sample dataset structure created!")
    print("Please add your plant disease images and masks to:")
    print("  - data/raw/images/{train,val,test}/")
    print("  - data/raw/masks/{train,val,test}/")
    print("\nMask format:")
    print("  - Binary images (0=healthy, 255=diseased)")
    print("  - Same filename as corresponding image")
    print("  - PNG format recommended")


def main():
    """Main function"""
    print("Plant Disease Segmentation - Sample Data Setup")
    print("=" * 50)
    
    # Create sample dataset structure
    create_sample_dataset()
    
    print("\nDataset preparation completed!")
    print("You can now train the model using:")
    print("  python main.py train")


if __name__ == "__main__":
    main()



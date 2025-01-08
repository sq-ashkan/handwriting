"""
Handwritten Character Recognition System

A deep learning-based OCR system for recognizing handwritten characters.

Author: Ashkan Sadri Ghamshi
Project: Deep Learning Character Recognition System
Course: HAWK University - Computer Science Department
Version: 1.0.0
Date: January 2025

This module is part of an academic project that implements a high-accuracy
Optical Character Recognition (OCR) system specialized in recognizing 
handwritten uppercase letters (A-Z) and digits (0-9).
"""

from pathlib import Path
import shutil
import logging
from tqdm import tqdm

# Base project path configuration
PROJECT_BASE = Path('/Users/roammer/Documents/Github/handwriting')
TEMP_PATH = PROJECT_BASE / 'data' / 'temp'
PROCESSED_PATH = PROJECT_BASE / 'data' / 'processed'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_merge(dataset_path: Path) -> bool:
    """
    Process and merge image files from dataset path to processed directory
    
    Args:
        dataset_path (Path): Path to the source dataset images
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure processed directory exists
        processed_images = PROCESSED_PATH / 'images'
        processed_images.mkdir(parents=True, exist_ok=True)

        # Handle documentation file
        doc_file = dataset_path.parent / 'documentation.txt'
        if doc_file.exists():
            dataset_name = dataset_path.parent.name
            with open(doc_file, 'r') as src_doc:
                content = src_doc.read()
                with open(PROCESSED_PATH / 'documentation.txt', 'a') as dest_doc:
                    dest_doc.write(content)
            logger.info(f"Appended documentation from {dataset_name}")

        # Get list of image files
        image_files = [img for img in Path(dataset_path).glob('*') if img.is_file()]
        
        # Copy images with progress bar
        for img in tqdm(image_files, 
                       desc=f"Copying {dataset_path.parent.name} images", 
                       unit='files', 
                       ncols=80):
            shutil.copy2(img, processed_images / img.name)
        
        logger.info(f"Successfully copied {len(image_files)} images from {dataset_path.parent.name}")
        return True
        
    except Exception as e:
        logger.error(f'Error processing {dataset_path.parent.name}: {str(e)}')
        return False

def merge() -> bool:
    """
    Main function to merge all temporary datasets into processed directory
    
    Returns:
        bool: True if all merges successful, False if any failed
    """
    success = True
    
    # Process each dataset in temp directory
    for dataset_dir in TEMP_PATH.iterdir():
        if dataset_dir.is_dir():
            images_path = dataset_dir / 'images'
            if images_path.exists():
                logger.info(f"Processing {dataset_dir.name} dataset")
                if not process_merge(images_path):
                    success = False
                    logger.error(f"Failed to process {dataset_dir.name}")
            else:
                logger.warning(f"No images directory found in {dataset_dir.name}")
    
    return success

if __name__ == "__main__":
    merge()
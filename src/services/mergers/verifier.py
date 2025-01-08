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
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def verify_image(dataset_path: Path, img_name: str) -> tuple[bool, Path]:
    """
    Verify if image exists and return its path
    Returns: (exists: bool, path: Path)
    """
    img_path = dataset_path / img_name
    if img_path.is_file():
        return True, img_path
        
    img_path_with_ext = dataset_path / f"{img_name}.png"
    if img_path_with_ext.is_file():
        return True, img_path_with_ext
        
    return False, img_path

def process_verify(dataset_path: Path) -> bool:
    try:
        # Setup paths
        final_path = Path('/Users/roammer/Documents/Github/handwriting/data/final')
        final_images = final_path / 'images'
        final_images.mkdir(parents=True, exist_ok=True)
        doc_path = Path(dataset_path).parent / 'documentation.txt'
        new_doc_path = final_path / 'documentation.txt'

        # Read and validate file content
        logger.info("Reading documentation file")
        valid_lines = []
        with open(doc_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    logger.warning(f"Line {line_num} invalid format: {line}")
                    continue
                    
                img_name = parts[0]
                label = parts[1]
                valid_lines.append((img_name, label))
        
        if not valid_lines:
            logger.error("No valid lines found in documentation file")
            return False
            
        # Verify all images exist
        logger.info("Verifying images")
        for img_name, _ in tqdm(valid_lines, desc="Verifying images"):
            exists, img_path = verify_image(dataset_path, img_name)
            if not exists:
                logger.error(f"Image not found: {img_name}")
                logger.error(f"Expected path: {img_path}")
                return False
        
        # Copy images and create new documentation
        logger.info("Copying images and creating new documentation")
        with open(new_doc_path, 'w') as f:
            for idx, (img_name, label) in tqdm(enumerate(valid_lines, 1), 
                                             total=len(valid_lines), 
                                             desc="Processing"):
                _, img_path = verify_image(dataset_path, img_name)
                
                new_name = f"Ash_PNG_{idx:06d}"
                new_path = final_images / f"{new_name}.png"
                shutil.copy2(img_path, new_path)
                
                f.write(f"{new_name} {label}\n")
        
        logger.info(f"Successfully processed {len(valid_lines)} images")
        return True
        
    except Exception as e:
        logger.error(f"Error in process_verify: {str(e)}")
        return False

def verify():
    """Main function to be called from main_modifier.py"""
    dataset_path = Path('/Users/roammer/Documents/Github/handwriting/data/processed/images')
    return process_verify(dataset_path)
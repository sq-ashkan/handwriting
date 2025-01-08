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

import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

class DataSplitter(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
    def _create_split_dirs(self) -> tuple:
        """Create directories for split datasets"""
        base_path = self.dataset_path.parent
        
        # Create directories
        train_dir = base_path / "train"
        val_dir = base_path / "val"
        test_dir = base_path / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(exist_ok=True)
            
        return train_dir, val_dir, test_dir
        
    def _copy_files(self, files: list, target_dir: Path) -> bool:
        """Copy files to target directory"""
        try:
            for file_path in files:
                shutil.copy2(file_path, target_dir)
            return True
        except Exception as e:
            logging.error(f"Error copying files to {target_dir}: {str(e)}")
            return False
            
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Not used in splitter"""
        return image
        
    def process(self) -> bool:
        """Override process method for splitting"""
        try:
            # Get all image files
            image_files = self._get_image_files()
            if not image_files:
                logging.error(f"No images found in {self.dataset_path}")
                return False
                
            # Create split directories
            train_dir, val_dir, test_dir = self._create_split_dirs()
            
            # Split files
            train_files, test_val_files = train_test_split(
                image_files,
                train_size=self.train_ratio,
                random_state=42
            )
            
            val_files, test_files = train_test_split(
                test_val_files,
                test_size=0.5,  # Split remaining data equally
                random_state=42
            )
            
            # Copy files to respective directories
            success = all([
                self._copy_files(train_files, train_dir),
                self._copy_files(val_files, val_dir),
                self._copy_files(test_files, test_dir)
            ])
            
            if success:
                logging.info(f"Dataset split completed: "
                           f"train={len(train_files)}, "
                           f"val={len(val_files)}, "
                           f"test={len(test_files)}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error splitting dataset: {str(e)}")
            return False
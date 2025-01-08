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

logger = logging.getLogger(__name__)

def cache_cleaner():
    """
    Clean up processed directory and __pycache__ folders
    """
    try:
        # Clean __pycache__ directories
        mergers_path = Path('src/services/mergers')
        pycache_folders = list(mergers_path.glob('**/__pycache__'))
        
        if pycache_folders:
            logger.info("Removing __pycache__ directories...")
            for cache_dir in pycache_folders:
                shutil.rmtree(cache_dir)
                logger.info(f"Removed cache directory: {cache_dir}")
        
        logger.info("Cache cleaning completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during cache cleaning: {e}")
        return False
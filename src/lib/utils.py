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

import sys
import logging
from pathlib import Path
from .constants import LOGS_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
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

# Root directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Dataset specific directories
ENGLISH_HANDWRITTEN_DIR = RAW_DIR / "english_handwritten"

MNIST_DIR = RAW_DIR / "mnist"
AZ_DIR = RAW_DIR / "az_handwritten"
CHARS74K_DIR = RAW_DIR / "chars74k"

# Create all directories
for dir_path in [DATA_DIR, LOGS_DIR, RAW_DIR, PROCESSED_DIR, 
                 ENGLISH_HANDWRITTEN_DIR, MNIST_DIR, AZ_DIR, CHARS74K_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
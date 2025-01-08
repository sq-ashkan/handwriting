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
from shutil import rmtree

def cleaner():
    base = Path('/Users/roammer/Documents/Github/handwriting/data')
    
    folders_to_clean = ['temp', 'raw', 'processed', 'final']
    for folder_name in folders_to_clean:
        folder = base / folder_name
        if folder.exists():
            rmtree(folder)
    
    ready = base / 'ready'
    if ready.exists():
        ready.rename(base / 'processed')
        print("✅ Folder 'ready' renamed to 'processed'.")
    else:
        print("Folder 'ready' not found.")

if __name__ == "__main__":
    cleaner()
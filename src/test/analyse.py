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

from abc import ABC, abstractmethod
import os
import json
import pathlib
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class DatasetAnalyzerInterface(ABC):
    @abstractmethod
    def analyze_dataset(self) -> Dict[str, Dict[str, int]]:
        pass

class DatasetPath:
    BASE_PATH = "/Users/roammer/Documents/Github/handwriting/data/processed"
    CATEGORIES = ['digits', 'lowercase', 'uppercase']
    
    @staticmethod
    def get_category_paths() -> Dict[str, str]:
        return {
            category: os.path.join(DatasetPath.BASE_PATH, category)
            for category in DatasetPath.CATEGORIES
        }

class ImageCounter:
    @staticmethod
    def count_images_in_folder(folder_path: str) -> int:
        """Count number of images in the images subdirectory of a folder"""
        images_path = os.path.join(folder_path, 'images')
        try:
            if not os.path.exists(images_path):
                return 0
            
            image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
            return sum(
                1 for f in os.listdir(images_path)
                if os.path.isfile(os.path.join(images_path, f))
                and os.path.splitext(f)[1].lower() in image_extensions
            )
        except Exception:
            return 0

    @staticmethod
    def count_character_images(category_path: str) -> Dict[str, int]:
        """Count images for each character in a category folder"""
        try:
            if not os.path.exists(category_path):
                return {}
            
            character_counts = {}
            for char_folder in os.listdir(category_path):
                char_path = os.path.join(category_path, char_folder)
                if os.path.isdir(char_path):
                    count = ImageCounter.count_images_in_folder(char_path)
                    if count > 0:  # Only include if images exist
                        character_counts[char_folder] = count
                        
            return character_counts
        except Exception:
            return {}

class DatasetAnalyzer(DatasetAnalyzerInterface):
    def analyze_dataset(self) -> Dict[str, Dict[str, int]]:
        category_paths = DatasetPath.get_category_paths()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda item: (item[0], ImageCounter.count_character_images(item[1])),
                category_paths.items()
            )
        
        # Combine results
        analysis_result = {}
        for category, counts in results:
            if counts:  # Only include categories with data
                analysis_result[category] = counts
                
        return analysis_result

class ResultWriter:
    @staticmethod
    def write_json(data: Dict[str, Dict[str, int]]) -> None:
        current_file_path = pathlib.Path(__file__).parent
        output_path = current_file_path / "analyse.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

def main():
    analyzer = DatasetAnalyzer()
    results = analyzer.analyze_dataset()
    ResultWriter.write_json(results)

if __name__ == "__main__":
    main()
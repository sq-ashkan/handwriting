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

import logging
from src.lib.cache_manager import CacheManager 
from src.services.processors.english_handwritten import EnglishHandwrittenProcessor
from src.services.processors.mnist_processor import MNISTProcessor
from src.services.processors.az_processor import AZProcessor
from src.services.processors.chars74k_processor import Chars74KProcessor

def setup_basic_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main() -> bool:
    try:
        # Setup logging first
        setup_basic_logging()
        
        processors = {
            "EH": EnglishHandwrittenProcessor(),
            "MNIST": MNISTProcessor(),
            "AZ": AZProcessor(),
            "Chars74K": Chars74KProcessor()
        }
        
        success = True
        for name, processor in processors.items():
            logging.info(f"Starting {name} dataset processing...")
            if not processor.process():
                logging.error(f"Failed to process {name} dataset")
                success = False
            else:
                logging.info(f"Successfully processed {name} dataset")
        
        return success
            
    except Exception as e:
        logging.error(f"Critical error in main processor: {str(e)}")
        return False
    finally:
        # Clean cache at the end
        CacheManager.cleanup()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
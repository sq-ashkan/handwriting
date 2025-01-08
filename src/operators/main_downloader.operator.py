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
from src.lib.utils import setup_logging
from src.lib.cache_manager import CacheManager
from src.services.downloaders.english_handwritten import EnglishHandwrittenDownloader
from src.services.downloaders.mnist_downloader import MNISTDatasetDownloader
from src.services.downloaders.az_downloader import AZDatasetDownloader
from src.services.downloaders.chars74k_downloader import Chars74KDatasetDownloader
from src.lib.constants import RAW_DIR

def main() -> bool:
    try:
        CacheManager.cleanup()
        setup_logging()
        
        downloaders = {
            "EH": EnglishHandwrittenDownloader(),
            "MNIST": MNISTDatasetDownloader(),
            "A-Z": AZDatasetDownloader(),
            "Chars74K": Chars74KDatasetDownloader()
        }
        
        success_status = {}
        
        for name, downloader in downloaders.items():
            logging.info(f"Starting {name} dataset download...")
            try:
                success_status[name] = downloader.run()
                if success_status[name]:
                    logging.info(f"{name} dataset successfully downloaded and processed")
                else:
                    logging.error(f"{name} dataset processing failed")
            except Exception as e:
                logging.error(f"Error with {name} dataset: {str(e)}")
                success_status[name] = False
        
        all_success = all(success_status.values())
        
        if all_success:
            logging.info("All datasets successfully downloaded and processed")
        else:
            failed = [name for name, success in success_status.items() if not success]
            logging.error(f"Failed datasets: {', '.join(failed)}")
        
        logging.info(f"\nDataset location: {RAW_DIR}")
        for name, success in success_status.items():
            status = "✓" if success else "✗"
            logging.info(f"{status} {name}")
        
        return all_success
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
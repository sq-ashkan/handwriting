import os
import sys
import logging
from pathlib import Path
import kagglehub
import shutil
import requests
from tqdm import tqdm

from src.lib.utils import create_directories
from src.lib.constants import IAM_DIR

class IAMDatasetDownloader:
    def __init__(self):
        """Initialize the downloader class"""
        self.logger = logging.getLogger(__name__)

    def download_with_progress(self, url: str, save_path: Path) -> None:
        """
        Download file with progress bar
        
        Args:
            url: File URL
            save_path: Save location path
        """
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as file, tqdm(
                desc=f"Downloading {save_path.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
                    
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            raise

    def download_dataset(self) -> Path:
        """
        Download dataset from Kaggle with progress tracking
        """
        self.logger.info("Starting IAM dataset download")
        try:
            with tqdm(total=100, desc="Downloading from Kaggle") as pbar:
                dataset_path = kagglehub.dataset_download(
                    "nibinv23/iam-handwriting-word-database"
                )
                pbar.update(100)
                
            dataset_path = Path(dataset_path)
            self.logger.info(f"Dataset downloaded to: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            raise

    def move_to_raw(self, temp_path: Path) -> bool:
        try:
            if not IAM_DIR.exists():
                IAM_DIR.mkdir(parents=True)
                
            target_image_dir = IAM_DIR / 'images'
            target_image_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Moving files to: {IAM_DIR}")
            
            # Move all PNG files to images directory
            png_files = list(temp_path.glob('**/*.png'))
            with tqdm(total=len(png_files), desc="Moving image files") as pbar:
                for img_file in png_files:
                    shutil.copy2(img_file, target_image_dir / img_file.name)
                    pbar.update(1)
            
            # Move documentation TXT file
            txt_files = list(temp_path.glob('**/*.txt'))
            if txt_files:
                shutil.copy2(txt_files[0], IAM_DIR / 'documentation.txt')
                
            return True
        except Exception as e:
            self.logger.error(f"File transfer failed: {str(e)}")
            return False

    def verify_download(self) -> bool:
        try:
            self.logger.info("Verifying download")
            
            images_dir = IAM_DIR / 'images'
            if not images_dir.exists():
                self.logger.error("Images directory not found")
                return False
                
            image_files = list(images_dir.glob('*.png'))
            if not image_files:
                self.logger.error("No image files found")
                return False
                
            if not (IAM_DIR / 'documentation.txt').exists():
                self.logger.error("Documentation file not found")
                return False
                
            self.logger.info(f"Found {len(image_files)} image files")
            return True
                
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False

    def run(self) -> bool:
        """
        Run the complete download process
        
        Returns:
            bool: Overall success status
        """
        try:
            create_directories()
            
            temp_path = self.download_dataset()
            if not self.move_to_raw(temp_path):
                return False
                
            if not self.verify_download():
                return False
                
            self.logger.info("Dataset download completed successfully")
            return True
                
        except Exception as e:
            self.logger.error(f"Download process failed: {str(e)}")
            return False
import os
import sys
import logging
from pathlib import Path
import kagglehub
import shutil
import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.lib.utils import create_directories
from src.lib.constants import ENGLISH_HANDWRITTEN_DIR

class EnglishHandwrittenDownloader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.target_dir = ENGLISH_HANDWRITTEN_DIR

    def download_dataset(self) -> Path:
        self.logger.info("Starting English Handwritten Characters dataset download")
        try:
            dataset_path = kagglehub.dataset_download(
                "dhruvildave/english-handwritten-characters-dataset"
            )
            return Path(dataset_path)
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            raise

    def create_documentation(self, csv_path: Path, image_dir: Path) -> None:
        try:
            df = pd.read_csv(csv_path)
            documentation_lines = []
            
            for _, row in df.iterrows():
                image_file = Path(row['image'])
                img_path = image_dir / image_file.name
                
                if img_path.exists():
                    with Image.open(img_path) as img:
                        w, h = img.size
                    
                    # Add EH prefix to image_id
                    img_id = f"EH_{image_file.stem}"
                    line = f"{img_id} 1 255 1 0 0 {w} {h} {img_id} {row['label']}"
                    documentation_lines.append(line)
                    
            with open(self.target_dir / 'documentation.txt', 'w') as f:
                f.write('\n'.join(documentation_lines))
                
        except Exception as e:
            self.logger.error(f"Failed to create documentation: {e}")
            raise

    def move_to_raw(self, temp_path: Path) -> bool:
        try:
            if not self.target_dir.exists():
                self.target_dir.mkdir(parents=True)
                
            target_image_dir = self.target_dir / 'images'
            target_image_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Moving files to: {self.target_dir}")
            
            # Move PNG files
            png_files = list(temp_path.glob('**/*.png'))
            with tqdm(total=len(png_files), desc="Moving image files") as pbar:
                for img_file in png_files:
                    shutil.copy2(img_file, target_image_dir / img_file.name)
                    pbar.update(1)
            
            # Process CSV to documentation.txt
            csv_files = list(temp_path.glob('**/*.csv'))
            if csv_files:
                self.create_documentation(csv_files[0], target_image_dir)
            else:
                self.logger.error("No CSV file found")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"File transfer failed: {str(e)}")
            return False

    def verify_download(self) -> bool:
        try:
            self.logger.info("Verifying download")
            
            images_dir = self.target_dir / 'images'
            if not images_dir.exists():
                self.logger.error("Images directory not found")
                return False
                
            image_files = list(images_dir.glob('*.png'))
            if not image_files:
                self.logger.error("No image files found")
                return False
                
            if not (self.target_dir / 'documentation.txt').exists():
                self.logger.error("Documentation file not found")
                return False
                
            self.logger.info(f"Found {len(image_files)} image files")
            return True
                
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return False

    def run(self) -> bool:
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
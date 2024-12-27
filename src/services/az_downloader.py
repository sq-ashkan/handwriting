from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
import shutil
import kagglehub
import pandas as pd
from src.lib.constants import AZ_DIR

class AZDatasetDownloader:
    def __init__(self):
        self.data_dir = AZ_DIR
        self.image_dir = AZ_DIR / 'images'
        self.doc_path = AZ_DIR / 'documentation.txt'
        self.kaggle_dataset = "sachinpatel21/az-handwritten-alphabets-in-csv-format"
        self.current_id = 1
        
    def _create_dirs(self):
        """Create necessary directories"""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_iam_id(self):
        """Generate IAM-compatible ID with az- prefix"""
        writer_id = f"az-a{self.current_id:02d}"
        doc_id = f"{self.current_id:03d}u"
        image_id = f"{writer_id}-{doc_id}-00-00"
        self.current_id += 1
        return image_id
        
    def _download_dataset(self):
        """Download dataset using kagglehub"""
        try:
            logging.info(f"Downloading {self.kaggle_dataset}...")
            dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
            logging.info(f"Dataset downloaded to: {dataset_path}")
            
            # Move downloaded files to our data directory
            csv_files = list(Path(dataset_path).glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV file found in downloaded dataset")
                
            source_file = csv_files[0]
            target_file = self.data_dir / source_file.name
            shutil.copy2(source_file, target_file)
            
            return target_file
            
        except Exception as e:
            logging.error(f"Failed to download dataset: {str(e)}")
            return None
            
    def _process_csv_to_images(self, csv_path):
        """Convert CSV data to images in IAM format"""
        records = []
        
        df = pd.read_csv(csv_path)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting CSV to images"):
            # Convert row to 28x28 image
            pixel_data = row.values[1:].reshape(28, 28)
            image = Image.fromarray(pixel_data.astype('uint8'))
            
            # Process image to match IAM format
            target_height = 27
            aspect_ratio = image.size[0] / image.size[1]
            target_width = int(target_height * aspect_ratio)
            target_width = min(max(target_width, 27), 51)
            processed_image = image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            
            # Generate IAM-style ID and save image
            image_id = self._generate_iam_id()
            save_path = self.image_dir / f"{image_id}.png"
            processed_image.save(str(save_path))
            
            # Create IAM-style documentation record
            x = 408 + (idx % 10) * 51
            y = 768 + (idx // 10) * 27
            w, h = processed_image.size
            char = chr(row.values[0] + ord('A'))  # Convert label to character
            
            record = f"{image_id} ok 154 1 {x} {y} {w} {h} AT {char}"
            records.append(record)
            
        return records
        
    def _create_documentation(self, records):
        """Create documentation file in IAM format"""
        with open(self.doc_path, 'w', encoding='utf-8') as f:
            f.write("# A-Z Handwritten Alphabets Database - IAM Format\n")
            f.write("# format: image_id ok graylevel components x y w h tag transcription\n\n")
            for record in records:
                f.write(f"{record}\n")
                
    def _cleanup_download(self, csv_path):
        """Remove temporary files after processing"""
        try:
            if csv_path and csv_path.exists():
                csv_path.unlink()
                logging.info(f"Cleaned up temporary file: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to clean up file {csv_path}: {str(e)}")
                    
    def run(self):
        """Main execution method"""
        try:
            self._create_dirs()
            logging.info("Starting A-Z Handwritten Alphabets download...")
            
            csv_path = self._download_dataset()
            if csv_path is None:
                return False
                
            records = self._process_csv_to_images(csv_path)
            self._create_documentation(records)
            
            self._cleanup_download(csv_path)
            
            logging.info("A-Z Handwritten Alphabets dataset downloaded and processed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return False

if __name__ == "__main__":
    from src.lib.utils import setup_logging
    setup_logging()
    downloader = AZDatasetDownloader()
    success = downloader.run()
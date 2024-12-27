from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging
import shutil
import kagglehub
import tarfile
import tempfile
from src.lib.constants import CHARS74K_DIR

class Chars74KDatasetDownloader:
    def __init__(self):
        self.data_dir = CHARS74K_DIR
        self.image_dir = CHARS74K_DIR / 'images'
        self.doc_path = CHARS74K_DIR / 'documentation.txt'
        self.kaggle_dataset = "supreethrao/chars74kdigitalenglishfont"
        self.current_id = 1
        self.folder_to_char_mapping = self._create_folder_mapping()
        
    def _create_folder_mapping(self):
        """Create mapping from Sample folders to characters"""
        mapping = {}
        
        # Numbers 0-9 (Sample001-Sample010)
        for i in range(10):
            folder_num = i + 1
            mapping[f"Sample{folder_num:03d}"] = str(i)
            
        # Uppercase letters A-Z (Sample011-Sample036)
        for i in range(26):
            folder_num = i + 11
            mapping[f"Sample{folder_num:03d}"] = chr(65 + i)  # ASCII 65 is 'A'
            
        # Lowercase letters a-z (Sample037-Sample062)
        for i in range(26):
            folder_num = i + 37
            mapping[f"Sample{folder_num:03d}"] = chr(97 + i)  # ASCII 97 is 'a'
            
        return mapping

    def _create_dirs(self):
        """Create necessary directories"""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_iam_id(self):
        """Generate IAM-compatible ID like ch-a01-001u-00-00"""
        writer_num = ((self.current_id - 1) // 100) + 1
        page_num = ((self.current_id - 1) % 100) + 1
        
        image_id = f"ch-a{writer_num:02d}-{page_num:03d}u-00-00"
        self.current_id += 1
        return image_id
        
    def _extract_tgz(self, tgz_path):
        temp_dir = Path(tempfile.mkdtemp())
        logging.info(f"Extracting to temporary directory: {temp_dir}")
        
        try:
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(path=temp_dir)
            logging.info("Extraction completed successfully")
            return temp_dir
        except Exception as e:
            logging.error(f"Failed to extract {tgz_path}: {str(e)}")
            return None
            
    def _download_dataset(self):
        try:
            logging.info(f"Downloading {self.kaggle_dataset}...")
            dataset_path = kagglehub.dataset_download(self.kaggle_dataset)
            dataset_path = Path(dataset_path)
            
            tgz_files = list(dataset_path.glob("*.tgz"))
            if not tgz_files:
                raise FileNotFoundError("No .tgz file found in downloaded dataset")
                
            extracted_path = self._extract_tgz(tgz_files[0])
            if extracted_path is None:
                return None
                
            return extracted_path
            
        except Exception as e:
            logging.error(f"Failed to download/extract dataset: {str(e)}")
            return None
            
    def _process_image(self, image_path):
        """Process image to standardized format"""
        with Image.open(image_path).convert('L') as img:
            target_size = (27, 27)
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            return resized_img
    
    def _process_dataset(self, dataset_path):
        records = []
        image_files = []
        
        # Find all Sample folders
        root_path = Path(dataset_path)
        sample_folders = sorted([
            d for d in root_path.rglob('Sample*')
            if d.is_dir() and d.name in self.folder_to_char_mapping
        ])
        
        # Collect images from each Sample folder
        for folder in sample_folders:
            char = self.folder_to_char_mapping[folder.name]
            patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            
            for pattern in patterns:
                found_files = list(folder.glob(pattern))
                image_files.extend((f, char) for f in found_files)
        
        if not image_files:
            logging.error(f"No image files found in {dataset_path}")
            return []
        
        logging.info(f"Processing {len(image_files)} images")
        
        for idx, (image_path, char) in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                processed_image = self._process_image(image_path)
                image_id = self._generate_iam_id()
                
                save_path = self.image_dir / f"{image_id}.png"
                processed_image.save(save_path)
                
                w, h = 27, 27
                grid_x = idx % 10
                grid_y = idx // 10
                x = 408 + grid_x * w
                y = 768 + grid_y * h
                
                record = f"{image_id} ok 154 1 {x} {y} {w} {h} AT {char}"
                records.append(record)
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                continue
                
        return records

    def _create_documentation(self, records):
        with open(self.doc_path, 'w', encoding='utf-8') as f:
            f.write("# Chars74K Database - IAM Format\n")
            f.write("# format: image_id ok graylevel components x y w h tag transcription\n\n")
            for record in records:
                f.write(f"{record}\n")

    def _cleanup_download(self, dataset_path):
        try:
            if dataset_path and dataset_path.exists():
                if dataset_path.is_dir():
                    shutil.rmtree(dataset_path)
                else:
                    dataset_path.unlink()
                logging.info(f"Cleaned up temporary path: {dataset_path}")
        except Exception as e:
            logging.error(f"Failed to clean up path {dataset_path}: {str(e)}")

    def run(self):
        try:
            self._create_dirs()
            logging.info("Starting Chars74K dataset download...")
            
            dataset_path = self._download_dataset()
            if dataset_path is None:
                return False
                
            records = self._process_dataset(dataset_path)
            if not records:
                logging.error("No images were processed successfully")
                return False
                
            self._create_documentation(records)
            self._cleanup_download(dataset_path)
            
            logging.info("Chars74K dataset processed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error in processing: {str(e)}")
            return False

if __name__ == "__main__":
    from src.lib.utils import setup_logging
    setup_logging()
    downloader = Chars74KDatasetDownloader()
    success = downloader.run()
from pathlib import Path
import shutil
import logging
from tqdm import tqdm
from src.services.enhancers.base_enhancer import BaseEnhancer

class Merger(BaseEnhancer):
    def __init__(self, dataset_path: Path):
        super().__init__(dataset_path)
        # Convert string path back to Path object
        self.dataset_path = Path(self.dataset_path)
        # Set processed path
        self.processed_path = Path('/Users/roammer/Documents/Github/handwriting/data/processed')
        self.processed_images = self.processed_path / 'images'
        self.processed_images.mkdir(parents=True, exist_ok=True)
    
    def _get_config(self) -> dict:
        return {}
    
    def process(self) -> bool:
        try:
            # Copy documentation file if exists
            doc_file = Path(self.dataset_path).parent / 'documentation.txt'
            if doc_file.exists():
                dataset_name = self.dataset_path.parent.name
                with open(doc_file, 'r') as src_doc:
                    content = src_doc.read()
                    with open(self.processed_path / 'documentation.txt', 'a') as dest_doc:
                        dest_doc.write(content)
                logging.info(f"Appended documentation from {doc_file}")
            
            # Get list of all image files first
            image_files = [img for img in Path(self.dataset_path).glob('*') if img.is_file()]
            
            # Copy all images with progress bar
            for img in tqdm(image_files, desc=f"Copying {self.dataset_path.parent.name} images", 
                          unit='files', ncols=80):
                shutil.copy2(img, self.processed_images / img.name)
            
            logging.info(f"Copied {len(image_files)} images to {self.processed_images}")
            return True
            
        except Exception as e:
            logging.error(f'Error copying files: {e}')
            return False
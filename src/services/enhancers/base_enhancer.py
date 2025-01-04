from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np
import cv2
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class BaseEnhancer(ABC):
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.cores = mp.cpu_count() - 1  # Reserve one core for system tasks
        
    def _get_image_files(self) -> List[Path]:
        """Get all image files from the dataset path"""
        return [f for f in self.dataset_path.glob("*") if f.is_file()]
        
    def _read_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Read image with error handling"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.error(f"Failed to read image: {image_path}")
                return None
            return image
        except Exception as e:
            logging.error(f"Error reading image {image_path}: {str(e)}")
            return None
            
    def _save_image(self, image: np.ndarray, image_path: Path) -> bool:
        """Save image with error handling"""
        try:
            cv2.imwrite(str(image_path), image)
            return True
        except Exception as e:
            logging.error(f"Error saving image {image_path}: {str(e)}")
            return False
            
    def _process_single_image(self, image_path: Path) -> bool:
        """Process a single image"""
        image = self._read_image(image_path)
        if image is None:
            return False
            
        try:
            enhanced_image = self._enhance_image(image)
            return self._save_image(enhanced_image, image_path)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return False
            
    def process(self) -> bool:
        """Process all images in parallel"""
        image_files = self._get_image_files()
        if not image_files:
            logging.error(f"No images found in {self.dataset_path}")
            return False
            
        with ProcessPoolExecutor(max_workers=self.cores) as executor:
            results = list(executor.map(self._process_single_image, image_files))
            
        success_rate = sum(results) / len(results)
        logging.info(f"Processing completed with {success_rate:.2%} success rate")
        return success_rate > 0.95  # Consider successful if 95% of images processed
        
    @abstractmethod
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Implement specific enhancement logic in derived classes"""
        pass
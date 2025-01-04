from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np
import cv2
from typing import List, Optional, Tuple
import multiprocessing as mp
from tqdm import tqdm

class BaseEnhancer(ABC):
    def __init__(self, dataset_path: Path):
        self.dataset_path = str(dataset_path)
        self.cores = mp.cpu_count() - 1
        
    @staticmethod
    def _process_image_static(args: Tuple[str, dict]) -> bool:
        """Static method for multiprocessing"""
        try:
            image_path, config = args
            image = cv2.imread(str(image_path))
            if image is None:
                logging.error(f"Failed to read image: {image_path}")
                return False
                
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Apply CLAHE with config params
            clahe = cv2.createCLAHE(
                clipLimit=float(config.get('clip_limit', 2.0)), 
                tileGridSize=(8,8)
            )
            normalized = clahe.apply(image)
            
            # Adjust mean and standard deviation
            current_mean = np.mean(normalized)
            current_std = np.std(normalized)
            target_mean = float(config.get('target_mean', 50.0))
            target_std = float(config.get('target_std', 70.0))
            
            # Scale and shift
            scaled = ((normalized - current_mean) * 
                     (target_std / current_std)) + target_mean
            
            enhanced = np.clip(scaled, 0, 255).astype(np.uint8)
            
            # Save the enhanced image
            cv2.imwrite(str(image_path), enhanced)
            return True
            
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return False
    
    def _get_image_files(self) -> List[str]:
        """Get all image files from the dataset path"""
        return [str(f) for f in Path(self.dataset_path).glob("*") if f.is_file()]
    
    def process(self) -> bool:
        """Process all images in parallel with progress bar"""
        image_files = self._get_image_files()
        if not image_files:
            logging.error(f"No images found in {self.dataset_path}")
            return False
        
        # Get config for the current dataset
        config = self._get_config()
        
        # Create args for multiprocessing
        process_args = [(img_path, config) for img_path in image_files]
        
        # Use Pool for multiprocessing
        with mp.Pool(processes=self.cores) as pool:
            results = list(tqdm(
                pool.imap(self._process_image_static, process_args),
                total=len(image_files),
                desc="Processing images",
                smoothing=0.1
            ))
        
        success_rate = sum(results) / len(results)
        logging.info(f"Processing completed with {success_rate:.2%} success rate")
        return success_rate > 0.95
    
    @abstractmethod
    def _get_config(self) -> dict:
        """Return configuration for the enhancement"""
        pass
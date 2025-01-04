from pathlib import Path
from src.services.enhancers.base_enhancer import BaseEnhancer
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

class BrightnessEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = Path(dataset_path).parent.name
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_config(self) -> dict:
        """Required by BaseEnhancer"""
        return {"threshold": 127}

    @staticmethod
    def _process_image_static(args: tuple) -> bool:
        """Static method for binary conversion"""
        try:
            image_path, config = args
            # Read image in grayscale mode
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return False

            # Apply binary thresholding
            _, binary = cv2.threshold(image, config['threshold'], 255, cv2.THRESH_BINARY)
            
            # Save the binary image
            cv2.imwrite(str(image_path), binary)
            return True
            
        except Exception as e:
            return False

    def process(self) -> bool:
        """Process all images with binary thresholding"""
        try:
            image_files = self._get_image_files()
            if not image_files:
                self.logger.error(f"No images found in {self.dataset_path}")
                return False

            config = self._get_config()
            process_args = [(img_path, config) for img_path in image_files]

            with ThreadPoolExecutor(max_workers=self.cores) as executor:
                results = list(executor.map(
                    self._process_image_static,
                    process_args
                ))

            success_rate = sum(results) / len(results)
            self.logger.info(f"Binary conversion completed with {success_rate:.2%} success rate")
            return success_rate > 0.95

        except Exception as e:
            self.logger.error(f"Binary conversion failed: {str(e)}")
            return False
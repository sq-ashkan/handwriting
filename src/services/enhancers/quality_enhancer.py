import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging
from skimage.measure import shannon_entropy

class QualityEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.configs = {
            "EH": {
                "min_entropy": 0.8,      # Target previous good value
                "contrast_limit": 40,
                "area_ratio": 0.05
            },
            "MNIST": {
                "min_entropy": 1.4,      # Target previous good value
                "contrast_limit": 60,
                "area_ratio": 0.08
            },
            "AZ": {
                "min_entropy": 0.9,      # Slightly increase
                "contrast_limit": 80,
                "area_ratio": 0.1
            },
            "Chars74K": {
                "min_entropy": 2.0,      # Target previous good value
                "contrast_limit": 90,
                "area_ratio": 0.12
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "min_entropy": 1.0,
            "contrast_limit": 70,
            "area_ratio": 0.08
        })

    def _enhance_entropy(self, image: np.ndarray, current_entropy: float) -> np.ndarray:
        config = self._get_config()
        
        if current_entropy < config["min_entropy"]:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
            
            # If still low entropy, try histogram equalization
            if shannon_entropy(enhanced) < config["min_entropy"]:
                return cv2.equalizeHist(enhanced)
            return enhanced
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        config = self._get_config()
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Find meaningful intensity range
        cumsum = np.cumsum(hist.ravel())
        total_pixels = cumsum[-1]
        
        # Find intensity values containing 1% and 99% of pixels
        low_cut = np.searchsorted(cumsum, total_pixels * 0.01)
        high_cut = np.searchsorted(cumsum, total_pixels * 0.99)
        
        # Ensure minimum contrast
        if high_cut - low_cut < config["contrast_limit"]:
            return cv2.equalizeHist(image)
            
        return np.clip((image - low_cut) * 255.0 / (high_cut - low_cut), 0, 255).astype(np.uint8)

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Calculate initial entropy
            current_entropy = shannon_entropy(image)
            
            # Enhance image quality
            enhanced = self._enhance_entropy(image, current_entropy)
            enhanced = self._enhance_contrast(enhanced)
            
            # Calculate final entropy
            final_entropy = shannon_entropy(enhanced)
            
            # Return enhanced image only if quality improved
            if final_entropy > current_entropy:
                return enhanced
                
            return image
            
        except Exception as e:
            logging.error(f"Error in quality enhancement: {str(e)}")
            return image
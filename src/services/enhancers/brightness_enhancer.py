import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging

class BrightnessEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.configs = {
            "EH": {
                "target_mean": 40.0,      # Increase from 13.29
                "target_std": 60.0,
                "clip_limit": 2.0
            },
            "MNIST": {
                "target_mean": 45.0,      # More balanced
                "target_std": 70.0,
                "clip_limit": 2.5
            },
            "AZ": {
                "target_mean": 50.0,      # Keep moderate brightness
                "target_std": 80.0,
                "clip_limit": 3.0
            },
            "Chars74K": {
                "target_mean": 60.0,      # Reduce from 87.45
                "target_std": 90.0,
                "clip_limit": 2.0
            }
        }
    
    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "target_mean": 50.0,
            "target_std": 70.0,
            "clip_limit": 2.0
        })

    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        config = self._get_config()
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=config["clip_limit"], 
            tileGridSize=(8,8)
        )
        normalized = clahe.apply(image)
        
        # Adjust mean and standard deviation
        current_mean = np.mean(normalized)
        current_std = np.std(normalized)
        
        # Scale and shift
        scaled = ((normalized - current_mean) * 
                 (config["target_std"] / current_std)) + config["target_mean"]
        
        return np.clip(scaled, 0, 255).astype(np.uint8)

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalize intensity
            enhanced = self._normalize_intensity(image)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in brightness enhancement: {str(e)}")
            return image
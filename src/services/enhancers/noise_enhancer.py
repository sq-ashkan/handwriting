import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging

class NoiseEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.configs = {
            "EH": {
                "noise_threshold": 2.5,
                "h_param": 5,          # Conservative denoising
                "bilateral_d": 7,
                "bilateral_sigma": 50
            },
            "MNIST": {
                "noise_threshold": 4.0,
                "h_param": 7,          # Moderate denoising
                "bilateral_d": 9,
                "bilateral_sigma": 75
            },
            "AZ": {
                "noise_threshold": 7.0,
                "h_param": 10,         # Aggressive denoising
                "bilateral_d": 11,
                "bilateral_sigma": 100
            },
            "Chars74K": {
                "noise_threshold": 5.0,
                "h_param": 8,
                "bilateral_d": 9,
                "bilateral_sigma": 75
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "noise_threshold": 5.0,
            "h_param": 7,
            "bilateral_d": 9,
            "bilateral_sigma": 75
        })

    def _remove_salt_pepper(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, 3)

    def _adaptive_denoising(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        config = self._get_config()
        
        if noise_level < config["noise_threshold"]:
            return cv2.bilateralFilter(
                image, 
                config["bilateral_d"],
                config["bilateral_sigma"],
                config["bilateral_sigma"]
            )
        else:
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=config["h_param"],
                templateWindowSize=7,
                searchWindowSize=21
            )

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Remove salt and pepper noise
            cleaned = self._remove_salt_pepper(image)
            
            # Estimate noise level
            noise_level = float(np.std(cv2.Laplacian(cleaned, cv2.CV_64F)))
            
            # Apply adaptive denoising
            denoised = self._adaptive_denoising(cleaned, noise_level)
            
            return denoised
            
        except Exception as e:
            logging.error(f"Error in noise reduction: {str(e)}")
            return image
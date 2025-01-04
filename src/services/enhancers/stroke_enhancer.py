import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging
from skimage import morphology

class StrokeEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.configs = {
            "EH": {
                "target_stroke_width": 1.2,    # Slight increase
                "max_scaling": 1.3,
                "min_scaling": 0.8
            },
            "MNIST": {
                "target_stroke_width": 1.3,    # Moderate increase
                "max_scaling": 1.4,
                "min_scaling": 0.7
            },
            "AZ": {
                "target_stroke_width": 1.4,    # Keep moderate
                "max_scaling": 1.5,
                "min_scaling": 0.7
            },
            "Chars74K": {
                "target_stroke_width": 1.6,    # Reduce significantly
                "max_scaling": 1.2,
                "min_scaling": 0.6
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "target_stroke_width": 1.4,
            "max_scaling": 1.4,
            "min_scaling": 0.7
        })

    def _calculate_stroke_width(self, binary_image: np.ndarray) -> float:
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        non_zero_distances = dist_transform[binary_image > 0]
        if len(non_zero_distances) == 0:
            return 0.0
        return float(np.mean(non_zero_distances) * 2)

    def _normalize_stroke_width(self, image: np.ndarray, current_width: float) -> np.ndarray:
        if current_width == 0:
            return image
            
        config = self._get_config()
        target_width = config["target_stroke_width"]
        
        # Calculate scaling factor with limits
        scale_factor = target_width / current_width
        scale_factor = np.clip(
            scale_factor,
            config["min_scaling"],
            config["max_scaling"]
        )
        
        if scale_factor > 1:
            kernel_size = int(np.ceil(scale_factor - 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.dilate(image, kernel, iterations=1)
        elif scale_factor < 1:
            kernel_size = int(np.ceil(1/scale_factor - 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            return cv2.erode(image, kernel, iterations=1)
            
        return image

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
            # Binarize image
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Calculate current stroke width
            current_width = self._calculate_stroke_width(binary)
            
            # Normalize stroke width
            enhanced = self._normalize_stroke_width(binary, current_width)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in stroke enhancement: {str(e)}")
            return image
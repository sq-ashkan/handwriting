import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class StrokeAnalysis:
    mean_width: float
    std_width: float
    local_widths: np.ndarray
    skeleton: np.ndarray
    direction_map: np.ndarray

class StrokeEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Enhanced configs with local adaptation parameters
        self.configs = {
            "EH": {
                "target_stroke_width": 1.2,
                "width_tolerance": 0.2,
                "max_scaling": 1.3,
                "min_scaling": 0.8,
                "skeleton_threshold": 0.6,
                "direction_smoothing": 0.7,
                "local_area_size": 5
            },
            "MNIST": {
                "target_stroke_width": 1.3,
                "width_tolerance": 0.3,
                "max_scaling": 1.4,
                "min_scaling": 0.7,
                "skeleton_threshold": 0.5,
                "direction_smoothing": 0.8,
                "local_area_size": 7
            },
            "AZ": {
                "target_stroke_width": 1.4,
                "width_tolerance": 0.25,
                "max_scaling": 1.5,
                "min_scaling": 0.7,
                "skeleton_threshold": 0.55,
                "direction_smoothing": 0.75,
                "local_area_size": 7
            },
            "Chars74K": {
                "target_stroke_width": 1.4,
                "width_tolerance": 0.4,
                "max_scaling": 1.0,
                "min_scaling": 0.6,
                "skeleton_threshold": 0.5,
                "direction_smoothing": 0.85,
                "local_area_size": 9
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "target_stroke_width": 1.4,
            "width_tolerance": 0.25,
            "max_scaling": 1.4,
            "min_scaling": 0.7,
            "skeleton_threshold": 0.55,
            "direction_smoothing": 0.8,
            "local_area_size": 7
        })

    def _create_direction_kernels(self) -> torch.Tensor:
        """Create directional kernels for stroke analysis"""
        kernels = []
        for angle in range(0, 180, 15):  # 12 directions
            theta = np.radians(angle)
            dx = np.cos(theta)
            dy = np.sin(theta)
            kernel = np.zeros((5, 5))
            
            # Create line kernel
            for i in range(5):
                x = int(2 + dx * (i - 2))
                y = int(2 + dy * (i - 2))
                if 0 <= x < 5 and 0 <= y < 5:
                    kernel[y, x] = 1
            
            kernels.append(torch.from_numpy(kernel).float().to(self.device))
        
        return torch.stack(kernels)

    def _analyze_stroke_properties(self, image: np.ndarray) -> StrokeAnalysis:
        """Analyze stroke properties using GPU acceleration"""
        try:
            # Convert to tensor
            img_tensor = torch.from_numpy(image).float().to(self.device)
            
            # Create binary image
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_tensor = torch.from_numpy(binary).float().to(self.device)
            
            # Calculate skeleton
            skeleton = morphology.skeletonize(binary > 0).astype(np.uint8) * 255
            skeleton_tensor = torch.from_numpy(skeleton).float().to(self.device)
            
            # Calculate distance transform
            dist_transform = distance_transform_edt(binary > 0)
            
            # Calculate local stroke widths
            local_widths = dist_transform * 2
            
            # Analyze stroke directions using convolution
            direction_kernels = self._create_direction_kernels()
            responses = F.conv2d(
                binary_tensor.unsqueeze(0).unsqueeze(0),
                direction_kernels.unsqueeze(1),
                padding=2
            )
            
            # Get dominant direction at each point
            direction_map = torch.argmax(responses.squeeze(), dim=0).cpu().numpy()
            
            return StrokeAnalysis(
                mean_width=float(np.mean(local_widths[skeleton > 0])),
                std_width=float(np.std(local_widths[skeleton > 0])),
                local_widths=local_widths,
                skeleton=skeleton,
                direction_map=direction_map
            )
            
        except Exception as e:
            logging.error(f"Error in stroke analysis: {str(e)}")
            return None

    def _calculate_local_scaling_factors(self, 
                                      stroke_analysis: StrokeAnalysis, 
                                      config: dict) -> np.ndarray:
        """Calculate scaling factors for each pixel based on local properties"""
        target_width = config["target_stroke_width"]
        local_widths = stroke_analysis.local_widths
        skeleton = stroke_analysis.skeleton
        
        # Initialize scaling factors
        scaling_factors = np.ones_like(local_widths, dtype=np.float32)
        
        # Calculate only for stroke pixels
        stroke_mask = skeleton > 0
        if np.any(stroke_mask):
            current_widths = local_widths[stroke_mask]
            
            # Calculate base scaling factors
            factors = target_width / (current_widths + 1e-6)
            
            # Apply limits
            factors = np.clip(factors, config["min_scaling"], config["max_scaling"])
            
            # Assign to output array
            scaling_factors[stroke_mask] = factors
            
            # Smooth scaling factors
            scaling_factors = cv2.GaussianBlur(
                scaling_factors,
                (config["local_area_size"], config["local_area_size"]),
                config["direction_smoothing"]
            )
        
        return scaling_factors

    def _apply_adaptive_morphology(self, 
                                 image: np.ndarray,
                                 scaling_factors: np.ndarray,
                                 direction_map: np.ndarray) -> np.ndarray:
        """Apply morphological operations adaptively based on local properties"""
        height, width = image.shape
        result = np.zeros_like(image)
        
        # Process in local windows
        window_size = 7
        pad = window_size // 2
        
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        for i in range(height):
            for j in range(width):
                if image[i, j] > 0:  # Process only stroke pixels
                    # Get local window
                    window = padded[i:i+window_size, j:j+window_size]
                    
                    # Get local scale and direction
                    scale = scaling_factors[i, j]
                    direction = direction_map[i, j] * 15  # Convert back to degrees
                    
                    # Create oriented kernel
                    kernel_size = max(3, int(round(scale * 3)))
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (kernel_size, kernel_size)
                    )
                    
                    if scale > 1:
                        # Dilation for thickening
                        processed = cv2.dilate(window, kernel)
                    else:
                        # Erosion for thinning
                        processed = cv2.erode(window, kernel)
                    
                    # Update center pixel
                    result[i, j] = processed[window_size//2, window_size//2]
        
        return result

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            config = self._get_config()
            
            # Analyze stroke properties
            stroke_analysis = self._analyze_stroke_properties(image)
            if stroke_analysis is None:
                return image
            
            # Calculate local scaling factors
            scaling_factors = self._calculate_local_scaling_factors(stroke_analysis, config)
            
            # Apply adaptive morphological operations
            enhanced = self._apply_adaptive_morphology(
                image,
                scaling_factors,
                stroke_analysis.direction_map
            )
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in stroke enhancement: {str(e)}")
            return image
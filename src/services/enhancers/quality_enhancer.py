"""
Handwritten Character Recognition System

A deep learning-based OCR system for recognizing handwritten characters.

Author: Ashkan Sadri Ghamshi
Project: Deep Learning Character Recognition System
Course: HAWK University - Computer Science Department
Version: 1.0.0
Date: January 2025

This module is part of an academic project that implements a high-accuracy
Optical Character Recognition (OCR) system specialized in recognizing 
handwritten uppercase letters (A-Z) and digits (0-9).
"""

import cv2
import numpy as np
from src.services.enhancers.base_enhancer import BaseEnhancer
import logging
from skimage.measure import shannon_entropy
from typing import Tuple
from dataclasses import dataclass
import torch
from scipy import stats

@dataclass
class CLAHEParams:
    clip_limit: float
    tile_size: Tuple[int, int]
    distribution: str

class QualityEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Enhanced configs with adaptive parameters
        self.configs = {
            "EH": {
                "min_entropy": 0.8,
                "contrast_limit": 40,
                "area_ratio": 0.05,
                "base_clip_limit": 2.0,
                "clip_limit_range": (1.5, 3.5),
                "tile_size_range": (6, 10),
                "distributions": ["uniform", "rayleigh"]
            },
            "MNIST": {
                "min_entropy": 1.4,
                "contrast_limit": 60,
                "area_ratio": 0.08,
                "base_clip_limit": 2.5,
                "clip_limit_range": (2.0, 4.0),
                "tile_size_range": (7, 11),
                "distributions": ["uniform", "exponential"]
            },
            "AZ": {
                "min_entropy": 0.9,
                "contrast_limit": 80,
                "area_ratio": 0.1,
                "base_clip_limit": 3.0,
                "clip_limit_range": (2.5, 4.5),
                "tile_size_range": (8, 12),
                "distributions": ["rayleigh", "uniform"]
            },
            "Chars74K": {
                "min_entropy": 2.0,
                "contrast_limit": 90,
                "area_ratio": 0.12,
                "base_clip_limit": 2.0,
                "clip_limit_range": (1.5, 3.5),
                "tile_size_range": (6, 10),
                "distributions": ["uniform", "exponential"]
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "min_entropy": 1.0,
            "contrast_limit": 70,
            "area_ratio": 0.08,
            "base_clip_limit": 2.5,
            "clip_limit_range": (2.0, 4.0),
            "tile_size_range": (7, 11),
            "distributions": ["uniform", "rayleigh"]
        })

    def _analyze_image_statistics(self, image: np.ndarray) -> dict:
        """Analyze image statistics for adaptive parameter selection"""
        try:
            # Convert to tensor for GPU acceleration
            img_tensor = torch.from_numpy(image).float().to(self.device)
            
            # Calculate basic statistics
            mean_val = float(torch.mean(img_tensor))
            std_val = float(torch.std(img_tensor))
            
            # Calculate histogram
            hist = torch.histc(img_tensor, bins=256, min=0, max=255)
            
            # Calculate entropy
            prob = hist / torch.sum(hist)
            entropy = float(-torch.sum(prob * torch.log2(prob + 1e-10)))
            
            # Calculate local contrast
            dx = img_tensor[1:, :] - img_tensor[:-1, :]
            dy = img_tensor[:, 1:] - img_tensor[:, :-1]
            gradient_magnitude = torch.sqrt(dx[:, :-1]**2 + dy[:-1, :]**2)
            local_contrast = float(torch.mean(gradient_magnitude))
            
            # Calculate skewness and kurtosis
            skewness = float(torch.mean((img_tensor - mean_val)**3) / (std_val**3))
            kurtosis = float(torch.mean((img_tensor - mean_val)**4) / (std_val**4))
            
            return {
                "mean": mean_val,
                "std": std_val,
                "entropy": entropy,
                "local_contrast": local_contrast,
                "skewness": skewness,
                "kurtosis": kurtosis
            }
            
        except Exception as e:
            logging.error(f"Error in image analysis: {str(e)}")
            return {
                "mean": np.mean(image),
                "std": np.std(image),
                "entropy": shannon_entropy(image),
                "local_contrast": 0,
                "skewness": stats.skew(image.ravel()),
                "kurtosis": stats.kurtosis(image.ravel())
            }

    def _determine_clahe_params(self, image: np.ndarray, stats: dict) -> CLAHEParams:
        """Determine optimal CLAHE parameters based on image statistics"""
        config = self._get_config()
        
        # Adjust clip limit based on local contrast and entropy
        base_clip = config["base_clip_limit"]
        clip_range = config["clip_limit_range"]
        
        # Higher clip limit for low contrast images
        contrast_factor = np.clip(1.0 - stats["local_contrast"] / 100, 0, 1)
        entropy_factor = np.clip(1.0 - stats["entropy"] / 8, 0, 1)
        
        clip_limit = base_clip + (clip_range[1] - clip_range[0]) * (
            0.6 * contrast_factor + 0.4 * entropy_factor
        )
        clip_limit = np.clip(clip_limit, clip_range[0], clip_range[1])
        
        # Adjust tile size based on image characteristics
        size_range = config["tile_size_range"]
        detail_factor = np.clip(stats["local_contrast"] / 100, 0, 1)
        
        tile_size = int(
            size_range[0] + (size_range[1] - size_range[0]) * (1 - detail_factor)
        )
        
        # Select distribution based on histogram characteristics
        if abs(stats["skewness"]) > 1 or stats["kurtosis"] > 3:
            distribution = config["distributions"][1]  # Use alternative distribution
        else:
            distribution = config["distributions"][0]  # Use primary distribution
            
        return CLAHEParams(
            clip_limit=float(clip_limit),
            tile_size=(tile_size, tile_size),
            distribution=distribution
        )

    def _apply_adaptive_clahe(self, image: np.ndarray, params: CLAHEParams) -> np.ndarray:
        """Apply CLAHE with adaptive parameters"""
        try:
            clahe = cv2.createCLAHE(
                clipLimit=params.clip_limit,
                tileGridSize=params.tile_size
            )
            
            if params.distribution == "rayleigh":
                # Apply Rayleigh distribution transformation
                alpha = np.sqrt(2 / np.pi)
                transformed = image * alpha
                enhanced = clahe.apply(transformed.astype(np.uint8))
                return (enhanced / alpha).astype(np.uint8)
                
            elif params.distribution == "exponential":
                # Apply exponential distribution transformation
                transformed = np.exp(image / 255) * 255
                enhanced = clahe.apply(transformed.astype(np.uint8))
                return (np.log(enhanced / 255) * 255).astype(np.uint8)
                
            else:  # uniform
                return clahe.apply(image)
                
        except Exception as e:
            logging.error(f"Error in CLAHE application: {str(e)}")
            return image

    def _enhance_entropy(self, image: np.ndarray, current_entropy: float) -> np.ndarray:
        """Enhance image entropy with adaptive CLAHE"""
        config = self._get_config()
        
        if current_entropy < config["min_entropy"]:
            # Analyze image statistics
            stats = self._analyze_image_statistics(image)
            
            # Determine optimal CLAHE parameters
            params = self._determine_clahe_params(image, stats)
            
            # Apply adaptive CLAHE
            enhanced = self._apply_adaptive_clahe(image, params)
            
            # If still low entropy, try alternative enhancement
            if shannon_entropy(enhanced) < config["min_entropy"]:
                # Try alternative parameters
                alt_params = CLAHEParams(
                    clip_limit=params.clip_limit * 1.5,
                    tile_size=(params.tile_size[0] - 2, params.tile_size[1] - 2),
                    distribution=config["distributions"][1]
                )
                return self._apply_adaptive_clahe(enhanced, alt_params)
                
            return enhanced
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        config = self._get_config()
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Find meaningful intensity range
        cumsum = np.cumsum(hist.ravel())
        total_pixels = cumsum[-1]
        
        # Find intensity values containing 5% and 95% of pixels (less aggressive)
        low_cut = np.searchsorted(cumsum, total_pixels * 0.05)
        high_cut = np.searchsorted(cumsum, total_pixels * 0.95)
        
        # Ensure minimum contrast
        if high_cut - low_cut < config["contrast_limit"]:
            return self._enhance_entropy(image, shannon_entropy(image))
            
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
            final_contrast = self._analyze_image_statistics(enhanced)["local_contrast"]
            
            # Return enhanced image only if both quality metrics improved
            if final_entropy > current_entropy and final_contrast > self._analyze_image_statistics(image)["local_contrast"]:
                return enhanced
                
            return image
            
        except Exception as e:
            logging.error(f"Error in quality enhancement: {str(e)}")
            return image
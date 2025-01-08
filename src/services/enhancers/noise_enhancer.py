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
from typing import Tuple
import torch
from torch.nn import functional as F

class NoiseEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = dataset_path.parent.name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.configs = {
            "EH": {
                "noise_threshold": 3.0,
                "bilateral_d": 5,
                "bilateral_sigma_color": 25,
                "bilateral_sigma_space": 25,
                "gaussian_kernel": 3,
                "gaussian_sigma": 0.3
            },
            "MNIST": {
                "noise_threshold": 4.5,
                "bilateral_d": 5,
                "bilateral_sigma_color": 30,
                "bilateral_sigma_space": 30,
                "gaussian_kernel": 3,
                "gaussian_sigma": 0.4
            },
            "AZ": {
                "noise_threshold": 8.0,
                "bilateral_d": 5,
                "bilateral_sigma_color": 35,
                "bilateral_sigma_space": 35,
                "gaussian_kernel": 3,
                "gaussian_sigma": 0.4
            },
            "Chars74K": {
                "noise_threshold": 5.5,
                "bilateral_d": 5,
                "bilateral_sigma_color": 30,
                "bilateral_sigma_space": 30,
                "gaussian_kernel": 3,
                "gaussian_sigma": 0.3
            }
        }

    def _get_config(self):
        return self.configs.get(self.dataset_name, {
            "noise_threshold": 4.0,
            "bilateral_d": 5,
            "bilateral_sigma_color": 30,
            "bilateral_sigma_space": 30,
            "gaussian_kernel": 3,
            "gaussian_sigma": 0.3
        })

    def _estimate_noise_type(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            img_tensor = torch.from_numpy(image).float().to(self.device)
            
            laplacian = torch.from_numpy(cv2.Laplacian(image, cv2.CV_64F)).to(self.device)
            noise_level = float(torch.std(laplacian))
            
            kernel = torch.ones(3, 3).to(self.device) / 9
            local_mean = F.conv2d(img_tensor.unsqueeze(0).unsqueeze(0), 
                                kernel.unsqueeze(0).unsqueeze(0), 
                                padding=1).squeeze()
            local_var = F.conv2d((img_tensor - local_mean).pow(2).unsqueeze(0).unsqueeze(0),
                               kernel.unsqueeze(0).unsqueeze(0),
                               padding=1).squeeze()
            
            if torch.max(local_var) / (torch.min(local_var) + 1e-6) > 15:
                return "impulse", noise_level
            else:
                return "gaussian", noise_level
                
        except Exception as e:
            logging.error(f"Error in noise estimation: {str(e)}")
            return "gaussian", 0.0

    def _remove_impulse_noise(self, image: np.ndarray) -> np.ndarray:
        try:
            denoised = cv2.medianBlur(image, 3)
            
            mask = np.abs(image.astype(float) - denoised.astype(float)) > 50
            result = image.copy()
            result[mask] = denoised[mask]
            
            return result
            
        except Exception as e:
            logging.error(f"Error in impulse noise removal: {str(e)}")
            return image

    def _minimal_gaussian_denoising(self, image: np.ndarray, noise_level: float, 
                                  config: dict) -> np.ndarray:
        try:
            if noise_level > config["noise_threshold"]:
                denoised = cv2.bilateralFilter(
                    image,
                    config["bilateral_d"],
                    config["bilateral_sigma_color"],
                    config["bilateral_sigma_space"]
                )
                return denoised
            return image
            
        except Exception as e:
            logging.error(f"Error in gaussian denoising: {str(e)}")
            return image

    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            config = self._get_config()
            
            noise_type, noise_level = self._estimate_noise_type(image)
            
            if noise_level < config["noise_threshold"] / 2:
                return image
            
            if noise_type == "impulse":
                enhanced = self._remove_impulse_noise(image)
            else:
                enhanced = self._minimal_gaussian_denoising(image, noise_level, config)
            
            enhanced_noise = cv2.Laplacian(enhanced, cv2.CV_64F).std()
            if enhanced_noise > noise_level or np.mean(np.abs(enhanced - image)) > 20:
                return image
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in image enhancement: {str(e)}")
            return image
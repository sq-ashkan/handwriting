# api/inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import os
from src.model import OptimizedOCR

class OCRPredictor:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU for Koyeb
        self.model = None
        self.idx_to_label = self._create_label_map()
        
    def _create_label_map(self):
        """Creates the label mapping dictionary"""
        label_map = {}
        # Add digits (0-9)
        for i in range(10):
            label_map[i] = str(i)
        # Add uppercase letters (A-Z)
        for i in range(26):
            label_map[i + 10] = chr(65 + i)
        return label_map
    
    async def _load_model(self):
        """Lazy loading of the model"""
        if self.model is None:
            try:
                self.model = OptimizedOCR(num_classes=36).to(self.device)
                model_path = 'best_model.pth'
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {model_path}")
                    
                # Load model with map_location to ensure CPU loading
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.model.eval()
                
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR"""
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Increase contrast
        image = ImageOps.autocontrast(image, cutoff=10)
        
        # Resize to 27x27 using Lanczos resampling
        image = image.resize((27, 27), Image.LANCZOS)
        
        return image
    
    async def _preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocesses the input image for model prediction"""
        try:
            # Open image using PIL
            image = Image.open(BytesIO(image_data))
            
            # Enhance image
            image = self._enhance_image(image)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize and binarize
            image_array = image_array.astype(np.float32) / 255.0
            threshold = np.mean(image_array)
            binary_image = (image_array > threshold).astype(np.float32)
            
            # Ensure dark background (invert if needed)
            if np.mean(binary_image) > 0.5:
                binary_image = 1.0 - binary_image
            
            # Add batch and channel dimensions
            image_tensor = torch.FloatTensor(binary_image).unsqueeze(0).unsqueeze(0)
            
            return image_tensor
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    async def predict_character(self, file) -> dict:
        """Predicts the character in the uploaded image"""
        try:
            # Load model if not loaded
            await self._load_model()
            
            # Read image data
            image_data = await file.read()
            
            # Preprocess image
            image_tensor = await self._preprocess_image(image_data)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = output.argmax(1).item()
                confidence = float(probabilities[0][predicted_idx])
                predicted_char = self.idx_to_label[predicted_idx]
            
            return {
                "character": predicted_char,
                "confidence": confidence,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
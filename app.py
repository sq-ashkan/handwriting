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

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import cv2
import os
import logging

# Configure logging for production environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HTML template for the root route documentation
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Character Recognition API - HAWK University</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #2c3e50;
        }
        .endpoint {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .method {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
        }
        .get {
            background: #61affe;
            color: white;
        }
        .post {
            background: #49cc90;
            color: white;
        }
        code {
            background: #272822;
            color: #f8f8f2;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 14px;
        }
        .header {
            border-bottom: 2px solid #eee;
            margin-bottom: 20px;
            padding-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Character Recognition API</h1>
            <p><strong>Author:</strong> Ashkan Sadri Ghamshi</p>
            <p><strong>Institution:</strong> HAWK University</p>
            <p><strong>Project:</strong> Deep Learning Character Recognition System</p>
            <p><strong>Date:</strong> Winter 2025</p>
        </div>

        <h2>API Documentation</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/ping</code>
            <p>Health check endpoint to verify if the service is running.</p>
            <p><strong>Response:</strong></p>
            <code>
                {
                    "status": "healthy",
                    "message": "OCR service is running"
                }
            </code>
        </div>

        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/predict</code>
            <p>Endpoint for character recognition. Accepts image file and returns predicted character.</p>
            <p><strong>Request:</strong> Form data with an image file</p>
            <p><strong>Response:</strong></p>
            <code>
                {
                    "character": "predicted_character",
                    "confidence": confidence_score,
                    "preprocessed_image": "base64_encoded_image",
                    "status": "success"
                }
            </code>
        </div>

        <h2>Model Information</h2>
        <ul>
            <li>Custom CNN architecture with attention mechanism</li>
            <li>Supports recognition of digits (0-9) and uppercase letters (A-Z)</li>
            <li>Input image is preprocessed and binarized for optimal recognition</li>
        </ul>

        <h2>Usage Notes</h2>
        <ul>
            <li>Images should be clear and contain a single character</li>
            <li>Supported image formats: JPEG, PNG</li>
            <li>For best results, use high-contrast images with dark text on light background</li>
        </ul>
    </div>
</body>
</html>
"""

app = Flask(__name__)

class AttentionModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = F.avg_pool2d(x, 1)
        attention = F.relu(self.conv1(attention))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = AttentionModule(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class OptimizedOCR(nn.Module):
    def __init__(self, num_classes: int = 36):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, stride=1, num_blocks=2)
        self.layer2 = self.make_layer(128, stride=2, num_blocks=2)
        self.layer3 = self.make_layer(256, stride=2, num_blocks=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def make_layer(self, out_channels: int, stride: int, num_blocks: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Global variables
device = torch.device("cpu")  # Using CPU for deployment
model = None
idx_to_label = None

def create_label_map():
    """Creates mapping for labels (0-9, A-Z)"""
    label_map = {}
    # Add digits (0-9)
    for i in range(10):
        label_map[i] = str(i)
    # Add uppercase letters (A-Z)
    for i in range(26):
        label_map[i + 10] = chr(65 + i)
    return label_map

def initialize_model():
    """Initialize model with pre-trained weights"""
    global model, idx_to_label
    
    try:
        # Create label mapping
        idx_to_label = create_label_map()
        
        # Initialize model
        model = OptimizedOCR(num_classes=36).to(device)
        
        # Check model file path
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load model with weights_only=True to avoid pickle warning
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Enhanced preprocessing function to convert input image to binary format"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Failed to load image")
        
        # Resize to 27x27
        image = cv2.resize(image, (27, 27), interpolation=cv2.INTER_AREA)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # C constant
        )
        
        # Check if image needs to be inverted
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        if white_pixels > total_pixels / 2:
            binary = cv2.bitwise_not(binary)
        
        # Convert to tensor and normalize
        image_tensor = torch.FloatTensor(binary).unsqueeze(0).unsqueeze(0) / 255.0
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    """Root route that displays API documentation"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "OCR service is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for character recognition"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        image_file = request.files['image']
        image_data = image_file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_data)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = output.argmax(1).item()
            confidence = float(probabilities[0][predicted_idx])
            predicted_char = idx_to_label[predicted_idx]
        
        # Convert tensor back to base64 encoded image for debugging
        preprocessed_img = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', preprocessed_img)
        preprocessed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "character": predicted_char,
            "confidence": confidence,
            "preprocessed_image": preprocessed_b64,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# Initialize model when starting
initialize_model()

# For WSGI deployment
application = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Added debug=True for development
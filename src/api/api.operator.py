from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import cv2

app = Flask(__name__)

# [Previous model architecture code remains the same...]
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
device = torch.device("mps")  # Using MPS for M2 Ultra
model = None
idx_to_label = None

def create_label_map():
    """Creates and returns the label mapping dictionary"""
    label_map = {}
    # Add digits (0-9)
    for i in range(10):
        label_map[i] = str(i)
    # Add uppercase letters (A-Z)
    for i in range(26):
        label_map[i + 10] = chr(65 + i)
    return label_map

def initialize_model():
    """Initializes the model with pre-trained weights"""
    global model, idx_to_label
    
    # Create label mapping
    idx_to_label = create_label_map()
    
    # Initialize model
    model = OptimizedOCR(num_classes=36).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    print("Model initialized successfully")

def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Enhanced preprocessing function to convert any input image to binary format"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Ensure image is loaded
    if image is None:
        raise ValueError("Failed to load image")
    
    # Resize to 27x27
    image = cv2.resize(image, (27, 27), interpolation=cv2.INTER_AREA)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    # Determine if we need to invert the image
    # Count white pixels
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    
    # If majority is white, invert the image
    if white_pixels > total_pixels / 2:
        binary = cv2.bitwise_not(binary)
    
    # Convert to tensor and normalize
    image_tensor = torch.FloatTensor(binary).unsqueeze(0).unsqueeze(0) / 255.0
    
    return image_tensor

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "OCR service is running"})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
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
        
        # For debugging, you can also return the preprocessed image
        # Convert tensor back to base64 encoded image
        preprocessed_img = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', preprocessed_img)
        preprocessed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "character": predicted_char,
            "confidence": confidence,
            "preprocessed_image": preprocessed_b64,  # This helps in debugging
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    initialize_model()
    app.run(host='0.0.0.0', port=5001)
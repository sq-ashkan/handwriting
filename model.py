import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import multiprocessing as mp
from typing import Tuple, List, Dict
import random

# Custom ResNet architecture optimized for 27x27 images
class CustomResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(CustomResNet, self).__init__()
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.layer1 = self.make_layer(32, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def make_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, max_samples: int = None, train: bool = True):
        self.root_dir = root_dir
        self.train = train
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Load digits
        digits_dir = os.path.join(root_dir, 'digits')
        for digit in sorted(os.listdir(digits_dir)):
            if os.path.isdir(os.path.join(digits_dir, digit)):
                self.label_to_idx[digit] = len(self.label_to_idx)
                self.idx_to_label[len(self.idx_to_label)] = digit
                
                img_dir = os.path.join(digits_dir, digit, 'images')
                img_files = os.listdir(img_dir)
                if max_samples:
                    img_files = random.sample(img_files, min(max_samples, len(img_files)))
                
                for img_file in img_files:
                    self.samples.append(os.path.join(img_dir, img_file))
                    self.labels.append(self.label_to_idx[digit])
        
        # Load uppercase letters
        uppercase_dir = os.path.join(root_dir, 'uppercase')
        for letter in sorted(os.listdir(uppercase_dir)):
            if os.path.isdir(os.path.join(uppercase_dir, letter)):
                self.label_to_idx[letter] = len(self.label_to_idx)
                self.idx_to_label[len(self.idx_to_label)] = letter
                
                img_dir = os.path.join(uppercase_dir, letter, 'images')
                img_files = os.listdir(img_dir)
                if max_samples:
                    img_files = random.sample(img_files, min(max_samples, len(img_files)))
                
                for img_file in img_files:
                    self.samples.append(os.path.join(img_dir, img_file))
                    self.labels.append(self.label_to_idx[letter])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = torch.FloatTensor(np.array(image)).unsqueeze(0) / 255.0
        label = self.labels[idx]
        
        return image, label

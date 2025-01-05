import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
import logging
import multiprocessing
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_valid_image(file_path: str) -> bool:
    """Check if the file is a valid image file."""
    return (file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and
            not file_path.startswith('.'))

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
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self.make_layer(64, stride=1, num_blocks=2)
        self.layer2 = self.make_layer(128, stride=2, num_blocks=2)
        self.layer3 = self.make_layer(256, stride=2, num_blocks=2)
        
        # Output layers
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

class OCRDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        
        categories = ['digits', 'uppercase']
        
        for category in categories:
            category_path = os.path.join(data_dir, category)
            if not os.path.exists(category_path):
                continue
                
            for char_folder in sorted(os.listdir(category_path)):
                char_path = os.path.join(category_path, char_folder)
                if not os.path.isdir(char_path):
                    continue
                    
                if char_folder not in self.label_to_idx:
                    self.label_to_idx[char_folder] = len(self.label_to_idx)
                
                images_path = os.path.join(char_path, 'images')
                image_files = [f for f in sorted(os.listdir(images_path))
                              if is_valid_image(f)]
                
                if max_samples:
                    image_files = image_files[:max_samples]
                
                for img_file in image_files:
                    self.samples.append(os.path.join(images_path, img_file))
                    self.labels.append(self.label_to_idx[char_folder])
        
        logging.info(f"Loaded {len(self.samples)} samples across {len(self.label_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert('L')
            image = torch.FloatTensor(np.array(image)).unsqueeze(0) / 255.0
            return image, self.labels[idx]
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise

def train(
    data_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 128,
    epochs: int = 50,
    initial_lr: float = 0.001,
    num_workers: int = 8
) -> nn.Module:
    
    # Device configuration - MPS for M2 Ultra
    device = torch.device("mps")
    torch.backends.cudnn.benchmark = True
    
    # Load and split dataset
    dataset = OCRDataset(data_dir, max_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model and move to device
    model = OptimizedOCR(num_classes=len(dataset.label_to_idx)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=initial_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        val_acc = 100. * val_correct / val_total
        logging.info(f'Epoch {epoch + 1} | '
                    f'Train Loss: {train_loss / len(train_loader):.3f} | '
                    f'Train Acc: {100. * train_correct / train_total:.2f}% | '
                    f'Val Loss: {val_loss / len(val_loader):.3f} | '
                    f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            logging.info(f'New best accuracy: {val_acc:.2f}%')
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
            if best_acc >= 99.0:
                logging.info(f'Reached target accuracy of 99%!')
                break
    
    return model

if __name__ == '__main__':
    # Set number of CPU threads for data loading
    num_workers = min(8, multiprocessing.cpu_count())
    
    # Start with smaller dataset
    logging.info("Starting initial training with 1000 samples per class...")
    model = train(
        data_dir='data/processed',
        max_samples=1000,
        batch_size=128,
        epochs=20,
        num_workers=num_workers
    )
    
    # If successful, train on full dataset
    logging.info("Starting full dataset training...")
    model = train(
        data_dir='data/processed',
        batch_size=128,
        epochs=50,
        num_workers=num_workers
    )
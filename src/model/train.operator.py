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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create plots directory
PLOTS_DIR = Path('training_plots')
PLOTS_DIR.mkdir(exist_ok=True)

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def save_training_progress(train_losses, train_accs, val_losses, val_accs, epochs, phase):
    """Save training and validation metrics plot."""
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss - Phase {phase}')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Training and Validation Accuracy - Phase {phase}')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'phase{phase}_training_progress_epoch_{epochs}.png')
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, epoch, phase):
    """Save confusion matrix visualization."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - Phase {phase} (Epoch {epoch})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'phase{phase}_confusion_matrix_epoch_{epoch}.png')
    plt.close()

def save_tsne_visualization(features, labels, class_names, epoch, phase):
    """Save t-SNE visualization of the learned features."""
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(15, 10))  
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab20', alpha=0.6, s=100) 
    
    # Calculate centroids for each class
    for idx, class_name in enumerate(class_names):
        mask = labels == idx
        if np.any(mask):
            centroid_x = features_2d[mask, 0].mean()
            centroid_y = features_2d[mask, 1].mean()
            plt.annotate(class_name, (centroid_x, centroid_y),
                        fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.legend(scatter.legend_elements()[0], 
              class_names,
              loc="center left",
              bbox_to_anchor=(1, 0.5),
              title="Classes")
    
    plt.title(f't-SNE Visualization - Phase {phase} (Epoch {epoch})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'phase{phase}_tsne_visualization_epoch_{epoch}.png', bbox_inches='tight')
    plt.close()

def save_learning_rate_plot(lr_history, epoch, phase):
    """Save learning rate changes plot."""
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history)
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule - Phase {phase} (Epoch {epoch})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'phase{phase}_learning_rate_epoch_{epoch}.png')
    plt.close()

def save_per_class_accuracy(class_correct, class_total, class_names, epoch, phase):
    """Save per-class accuracy plot."""
    accuracies = [100 * correct / total for correct, total in zip(class_correct, class_total)]
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(class_names, accuracies)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Per-Class Accuracy - Phase {phase} (Epoch {epoch})')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'phase{phase}_per_class_accuracy_epoch_{epoch}.png')
    plt.close()

def should_visualize(epoch: int) -> bool:
    """Determine if visualization should be performed for this epoch."""
    return epoch == 1 or epoch % 10 == 0

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
    num_workers: int = 8,
    phase: int = 1
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
    
    # Initialize tracking variables
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lr_history = []
    
    # Get class names
    idx_to_label = {v: k for k, v in dataset.label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(dataset.label_to_idx))]
    
    # Initialize per-class tracking
    class_correct = [0] * len(dataset.label_to_idx)
    class_total = [0] * len(dataset.label_to_idx)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Phase {phase} - Epoch {epoch + 1}/{epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track learning rate
            lr_history.append(optimizer.param_groups[0]['lr'])
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        features_list = []
        labels_list = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Phase {phase} - Epoch {epoch + 1}/{epochs} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Only extract features if we're going to visualize this epoch
                if should_visualize(epoch + 1):
                    features = model.avg_pool(model.layer3(model.layer2(
                        model.layer1(F.relu(model.bn1(model.conv1(inputs))))))).view(inputs.size(0), -1)
                    features_list.append(features.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Track per-class accuracy
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Log performance
        logging.info(f'Phase {phase} - Epoch {epoch + 1} | '
                    f'Train Loss: {epoch_train_loss:.3f} | '
                    f'Train Acc: {epoch_train_acc:.2f}% | '
                    f'Val Loss: {epoch_val_loss:.3f} | '
                    f'Val Acc: {epoch_val_acc:.2f}%')
        
        # Save plots only on specific epochs
        if should_visualize(epoch + 1):
            logging.info(f'Generating visualizations for phase {phase}, epoch {epoch + 1}...')
            save_training_progress(train_losses, train_accs, val_losses, val_accs, epoch + 1, phase)
            if features_list and labels_list:
                features_array = np.concatenate(features_list)
                labels_array = np.concatenate(labels_list)
                save_tsne_visualization(features_array, labels_array, class_names, epoch + 1, phase)
            save_learning_rate_plot(lr_history, epoch + 1, phase)
            save_per_class_accuracy(class_correct, class_total, class_names, epoch + 1, phase)
            save_confusion_matrix(all_labels, all_predictions, class_names, epoch + 1, phase)
        
        # Save best model
        if epoch_val_acc > best_acc:
            logging.info(f'Phase {phase} - New best accuracy: {epoch_val_acc:.2f}%')
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f'best_model_phase{phase}.pth')
            
            if best_acc >= 99.0:
                logging.info(f'Phase {phase} - Reached target accuracy of 99%!')
                break
    
    return model

if __name__ == '__main__':
    # Set number of CPU threads for data loading
    num_workers = min(8, multiprocessing.cpu_count())
    
    # Phase 1: Initial training with limited data
    logging.info("Starting phase 1: initial training with 1000 samples per class...")
    model = train(
        data_dir='data/processed',
        max_samples=1000,
        batch_size=128,
        epochs=20,
        num_workers=num_workers,
        phase=1  # Specify phase 1
    )
    
    # Phase 2: Full dataset training
    logging.info("Starting phase 2: full dataset training...")
    model = train(
        data_dir='data/processed',
        batch_size=128,
        epochs=50,
        num_workers=num_workers,
        phase=2  # Specify phase 2
    )
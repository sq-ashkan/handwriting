import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from datetime import datetime
import random
import shutil
import logging
from pathlib import Path
import glob

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_count(folder_path: str) -> int:
    """Get exact count of valid image files in folder."""
    pattern = os.path.join(folder_path, "*.[pjJ][npNP][gG]*")
    return len(glob.glob(pattern))

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load and preprocess image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 127:
            gray = 255 - gray
        return gray
    except Exception:
        return None

def apply_augmentation(image: np.ndarray, aug_type: str, value: float) -> np.ndarray:
    """Apply augmentation transformation."""
    if aug_type == "ROT":
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, value, 1.0)
        return cv2.warpAffine(image, matrix, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    else:  # SCL
        h, w = image.shape[:2]
        new_h = int(h * value)
        new_w = int(w * value)
        scaled = cv2.resize(image, (new_w, new_h))
        result = np.zeros((h, w), dtype=np.uint8)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        if y_offset >= 0 and x_offset >= 0:
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        else:
            y_start = abs(min(y_offset, 0))
            x_start = abs(min(x_offset, 0))
            result = scaled[y_start:y_start+h, x_start:x_start+w]
        return result

def generate_augmentations(needed: int) -> List[Tuple[str, float]]:
    """Generate augmentation parameters ensuring enough variations."""
    augmentations = []
    
    # More granular rotation angles
    angles = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # More granular scaling factors
    scales = [0.995, 0.9965, 0.998, 0.9995, 1.0005, 1.002, 1.0035, 1.005]
    
    # Calculate repetitions needed
    total_variations = len(angles) + len(scales)
    repetitions = (needed // total_variations) + 1
    
    for _ in range(repetitions):
        for angle in angles:
            augmentations.append(("ROT", angle))
        for scale in scales:
            augmentations.append(("SCL", scale))
    
    random.shuffle(augmentations)
    return augmentations[:needed]

def process_character_folder(char_folder: str, target_count: int = 20000) -> None:
    """Process single character folder ensuring exact target count."""
    images_path = os.path.join(char_folder, 'images')
    if not os.path.exists(images_path):
        logging.error(f"Images path does not exist: {images_path}")
        return

    current_count = get_image_count(images_path)
    if current_count == target_count:
        logging.info(f"Folder {os.path.basename(char_folder)} already has {target_count} images.")
        return

    if current_count > target_count:
        backup_path = os.path.join(os.path.dirname(char_folder), 
                                 f'backup_{datetime.now().strftime("%Y%m%d_%H%M")}')
        os.makedirs(backup_path, exist_ok=True)
        
        all_images = glob.glob(os.path.join(images_path, "*.[pjJ][npNP][gG]*"))
        to_move = random.sample(all_images, current_count - target_count)
        
        for img_path in tqdm(to_move, desc="Moving excess images"):
            shutil.move(img_path, os.path.join(backup_path, os.path.basename(img_path)))
        return

    # Load base images
    base_images = []
    all_images = glob.glob(os.path.join(images_path, "*.[pjJ][npNP][gG]*"))
    
    for img_path in tqdm(all_images, desc="Loading base images"):
        img = load_image(img_path)
        if img is not None:
            base_images.append((img, os.path.splitext(os.path.basename(img_path))[0]))

    if not base_images:
        logging.error(f"No valid base images found in {images_path}")
        return

    needed = target_count - current_count
    augmentations = generate_augmentations(needed)
    unique_identifier = datetime.now().strftime("%H%M%S")
    generated = 0
    
    with tqdm(total=needed, desc=f"Generating images for {os.path.basename(char_folder)}") as pbar:
        for idx, (aug_type, value) in enumerate(augmentations):
            if generated >= needed:
                break
                
            base_img, base_name = random.choice(base_images)
            aug_img = apply_augmentation(base_img, aug_type, value)
            
            filename = f"{base_name}_{aug_type}_{int(abs(value*1000)):04d}_{unique_identifier}_{idx:06d}.png"
            save_path = os.path.join(images_path, filename)
            
            if cv2.imwrite(save_path, aug_img):
                generated += 1
                pbar.update(1)

    final_count = get_image_count(images_path)
    if final_count < target_count:
        remaining = target_count - final_count
        logging.info(f"Generating remaining {remaining} images...")
        process_character_folder(char_folder, target_count)
    elif final_count > target_count:
        excess = final_count - target_count
        logging.info(f"Removing {excess} excess images...")
        all_images = glob.glob(os.path.join(images_path, "*.[pjJ][npNP][gG]*"))
        to_remove = random.sample(all_images, excess)
        for img_path in to_remove:
            os.remove(img_path)

def balance_dataset(base_path: str) -> None:
    """Balance entire dataset to exactly 20,000 images per character."""
    categories = ['digits', 'uppercase']  # Only digits and uppercase
    target_count = 20000
    
    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            logging.error(f"Category path does not exist: {category_path}")
            continue

        char_folders = sorted([f for f in os.listdir(category_path) 
                             if os.path.isdir(os.path.join(category_path, f))])
        
        for char_folder in tqdm(char_folders, desc=f"Processing {category}", leave=False):
            full_path = os.path.join(category_path, char_folder)
            process_character_folder(full_path, target_count)

def main():
    base_path = "/Users/roammer/Documents/Github/handwriting/data/processed"
    logging.info("Starting dataset balancing...")
    balance_dataset(base_path)
    logging.info("Dataset balancing completed!")

if __name__ == "__main__":
    main()
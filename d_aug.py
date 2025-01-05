import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
import random
import shutil

def create_backup(folder_path: str) -> str:
    """Create backup folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    backup_path = os.path.join(os.path.dirname(folder_path), f'backup_{timestamp}')
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    return backup_path

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load and preprocess image."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Apply rotation while maintaining image dimensions."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Add border to prevent cutting off parts of the image
    border = int(max(w, h) * 0.1)
    padded = cv2.copyMakeBorder(image, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=255)
    
    # Adjust rotation matrix for the new size
    p_h, p_w = padded.shape[:2]
    p_center = (p_w // 2, p_h // 2)
    matrix = cv2.getRotationMatrix2D(p_center, angle, 1.0)
    
    rotated = cv2.warpAffine(padded, matrix, (p_w, p_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    
    # Crop back to original size
    y_start = (p_h - h) // 2
    x_start = (p_w - w) // 2
    return rotated[y_start:y_start+h, x_start:x_start+w]

def apply_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """Apply scaling while maintaining image dimensions."""
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize the image
    scaled = cv2.resize(image, (new_w, new_h))
    
    # Create output image filled with white
    result = np.full((h, w), 255, dtype=np.uint8)
    
    # Calculate offsets for centering
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2
    
    # Copy the scaled image to the center of the result
    if scale < 1.0:
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
    else:
        # If scaled up, take the center portion
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        result = scaled[y_start:y_start+h, x_start:x_start+w]
    
    return result

def apply_elastic_transform(image: np.ndarray) -> np.ndarray:
    """Apply gentle elastic transformation."""
    h, w = image.shape[:2]
    alpha = 300  # Controls intensity of deformation
    sigma = 10   # Controls smoothness of deformation
    
    # Generate random displacement fields
    dx = np.random.rand(h, w) * 2 - 1
    dy = np.random.rand(h, w) * 2 - 1
    
    # Smooth the displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    
    # Create mesh grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Add displacement fields to the mesh grid
    mapx = np.float32(x + dx)
    mapy = np.float32(y + dy)
    
    # Apply the transformation
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255)

def apply_augmentation(image: np.ndarray, base_filename: str) -> List[Tuple[np.ndarray, str]]:
    """Apply various augmentation techniques."""
    augmented = []
    
    # Rotation augmentation
    angles = [random.uniform(1, 6) * random.choice([-1, 1]) for _ in range(3)]
    for angle in angles:
        rotated = apply_rotation(image, angle)
        augmented.append((rotated, f"{base_filename}_ROT_{int(abs(angle)):+03d}"))
    
    # Scale augmentation
    scales = [0.95, 1.05]
    for scale in scales:
        scaled = apply_scale(image, scale)
        augmented.append((scaled, f"{base_filename}_SCL_{int(scale*100):03d}"))
    
    # Elastic transformation
    elastic = apply_elastic_transform(image)
    augmented.append((elastic, f"{base_filename}_ELS_001"))
    
    return augmented

def process_character_folder(char_folder: str, target_count: int) -> None:
    """Process a single character folder."""
    images_path = os.path.join(char_folder, 'images')
    if not os.path.exists(images_path):
        print(f"Images path does not exist: {images_path}")
        return

    # Get existing images
    existing_images = [f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(existing_images)
    
    if current_count == target_count:
        return

    if current_count > target_count:
        # Move excess images to backup
        backup_path = create_backup(images_path)
        images_to_move = random.sample(existing_images, current_count - target_count)
        
        print(f"\nMoving {len(images_to_move)} images to backup for {os.path.basename(char_folder)}")
        for img in tqdm(images_to_move, desc="Backing up excess images"):
            src = os.path.join(images_path, img)
            dst = os.path.join(backup_path, img)
            shutil.move(src, dst)
            
        existing_images = [f for f in existing_images if f not in set(images_to_move)]
    
    if len(existing_images) < target_count:
        needed = target_count - len(existing_images)
        
        # Load and prepare base images
        print(f"\nPreparing base images for {os.path.basename(char_folder)}")
        base_images = []
        for img_path in tqdm(existing_images, desc="Loading images"):
            full_path = os.path.join(images_path, img_path)
            img = load_image(full_path)
            if img is not None:
                base_images.append((img, os.path.splitext(img_path)[0]))
        
        if not base_images:
            print(f"No valid images found in {images_path}")
            return
        
        # Generate augmented images
        print(f"\nGenerating {needed} new images for {os.path.basename(char_folder)}")
        generated = 0
        with tqdm(total=needed, desc="Generating images") as pbar:
            while generated < needed:
                base_img, base_name = random.choice(base_images)
                new_images = apply_augmentation(base_img, base_name)
                
                for img, filename in new_images:
                    if generated < needed:
                        save_path = os.path.join(images_path, f"{filename}.png")
                        if cv2.imwrite(save_path, img):
                            generated += 1
                            pbar.update(1)
                    else:
                        break

def balance_dataset(base_path: str, target_counts: Dict[str, Dict[str, int]]) -> None:
    """Balance the entire dataset."""
    categories = ['digits', 'lowercase', 'uppercase']
    
    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(base_path, category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        char_folders = [f for f in os.listdir(category_path) 
                       if os.path.isdir(os.path.join(category_path, f))]
        
        for char_folder in tqdm(char_folders, desc=f"Processing {category}", leave=False):
            full_path = os.path.join(category_path, char_folder)
            target = target_counts.get(category, {}).get(char_folder, 0)
            if target > 0:
                process_character_folder(full_path, target)

def main():
    base_path = "/Users/roammer/Documents/Github/handwriting/data/processed"
    
    target_counts = {
        "digits": {str(i): 8000 for i in range(10)},
        "lowercase": {chr(i): 20000 for i in range(ord('a'), ord('z')+1)},
        "uppercase": {chr(i): 20000 for i in range(ord('A'), ord('Z')+1)}
    }
    
    print("Starting dataset balancing...")
    balance_dataset(base_path, target_counts)
    print("Dataset balancing completed!")

if __name__ == "__main__":
    main()
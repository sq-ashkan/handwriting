import logging
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

class EnglishHandwrittenProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "english_handwritten"
        self.temp_path = self.project_root / "data" / "temp" / "EH"
        self.target_size = (27, 27)
        self.target_ratio = (0.6, 0.7)
        self.device = torch.device("mps")
        self.num_workers = min(mp.cpu_count() - 1, 16)
        self.chunk_size = 256
        self.cache_dir = self.project_root / "data" / "cache" / "EH"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self) -> bool:
        try:
            self.logger.info("Starting English handwritten text processing...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False

            progress = 0
            self._parallel_copy_dataset()
            progress += 25
            self.logger.info(f"Processing progress: {progress}%")
            
            image_files = list((self.temp_path / "images").glob("*.png"))
            chunks = [image_files[i:i + self.chunk_size] for i in range(0, len(image_files), self.chunk_size)]
            
            processed_chunks = 0
            total_chunks = len(chunks)
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_chunk, chunk)
                    futures.append(future)
                
                for future in tqdm(futures, total=len(futures), desc="Processing image chunks"):
                    future.result()
                    processed_chunks += 1
                    current_progress = 25 + (processed_chunks / total_chunks * 50)
                    self.logger.info(f"Processing progress: {int(current_progress)}%")
            
            self._validate_sample_image()
            self.logger.info("Processing progress: 100%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}")
            return False
            
    def _process_chunk(self, image_paths: List[Path]) -> None:
        try:
            processed_images = []
            valid_paths = []
            
            for img_path in image_paths:
                try:
                    cache_path = self.cache_dir / img_path.name
                    if cache_path.exists():
                        shutil.copy2(cache_path, img_path)
                        continue
                    
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        processed = self._process_single_image(image)
                        if processed is not None:
                            processed_images.append(processed)
                            valid_paths.append(img_path)
                            cv2.imwrite(str(cache_path), processed)
                except Exception as e:
                    self.logger.warning(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            if not processed_images:
                return
                
            batch = torch.tensor(np.stack(processed_images)).to(self.device)
            processed_batch = self._process_tensor_batch(batch)
            
            for img, path in zip(processed_batch.cpu().numpy(), valid_paths):
                cv2.imwrite(str(path), img)
                
        except Exception as e:
            self.logger.error(f"Error in chunk processing: {str(e)}")

    def _process_single_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255.0
            
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
            
            image_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            image = clahe.apply(image_uint8).astype(np.float32) / 255.0
            
            image_uint8 = (image * 255).astype(np.uint8)
            binary = cv2.adaptiveThreshold(
                image_uint8, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            
            binary = self._clean_artifacts(binary)
            processed = self._center_and_adjust_ratio(binary)
            
            if processed is None:
                return None
                
            return self._ensure_margins(processed)
            
        except Exception as e:
            self.logger.warning(f"Error in single image processing: {str(e)}")
            return None

    def _clean_artifacts(self, image: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        min_size = 5
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                image[labels == i] = 0
        
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image

    def _center_and_adjust_ratio(self, image: np.ndarray) -> Optional[np.ndarray]:
        coords = cv2.findNonZero(image)
        if coords is None:
            return None
            
        x, y, w, h = cv2.boundingRect(coords)
        current_ratio = (w * h) / (self.target_size[0] * self.target_size[1])
        
        if current_ratio < self.target_ratio[0]:
            scale = np.sqrt(self.target_ratio[0] / current_ratio)
        elif current_ratio > self.target_ratio[1]:
            scale = np.sqrt(self.target_ratio[1] / current_ratio)
        else:
            scale = 1.0
            
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > self.target_size[0] - 4 or new_h > self.target_size[1] - 4:
            scale = min((self.target_size[0] - 4) / w, (self.target_size[1] - 4) / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        
        char_img = image[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        result = np.zeros(self.target_size, dtype=np.uint8)
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_img
        
        return result

    def _ensure_margins(self, image: np.ndarray) -> np.ndarray:
        min_margin = 2
        coords = cv2.findNonZero(image)
        if coords is None:
            return image
            
        x, y, w, h = cv2.boundingRect(coords)
        
        if x < min_margin or y < min_margin or \
           x + w > image.shape[1] - min_margin or \
           y + h > image.shape[0] - min_margin:
            new_image = np.zeros_like(image)
            new_w = min(w, image.shape[1] - 2*min_margin)
            new_h = min(h, image.shape[0] - 2*min_margin)
            char_img = image[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x_offset = (image.shape[1] - new_w) // 2
            y_offset = (image.shape[0] - new_h) // 2
            new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_img
            return new_image
        
        return image

    @torch.no_grad()
    def _process_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        try:
            kernel = torch.ones((2, 2), device=self.device)
            batch = F.conv2d(
                batch.float().unsqueeze(1),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze(1)
            return (batch > 0).byte() * 255
        except Exception as e:
            self.logger.error(f"Error in tensor batch processing: {str(e)}")
            return batch.byte()

    def _validate_sample_image(self) -> None:
        try:
            sample_path = next((self.temp_path / "images").glob("*.png"))
            image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            
            assert image.shape == self.target_size, f"Invalid dimensions: {image.shape}"
            
            coords = cv2.findNonZero(image)
            x, y, w, h = cv2.boundingRect(coords)
            ratio = (w * h) / (image.shape[0] * image.shape[1])
            assert self.target_ratio[0] <= ratio <= self.target_ratio[1], \
                   f"Invalid occupation ratio: {ratio}"
            
            min_margin = 2
            assert x >= min_margin and y >= min_margin, "Insufficient margins"
            assert x + w <= image.shape[1] - min_margin, "Insufficient margins"
            assert y + h <= image.shape[0] - min_margin, "Insufficient margins"
            
            self.logger.info("Sample image validation successful")
        except Exception as e:
            self.logger.error(f"Sample validation failed: {str(e)}")

    def _parallel_copy_dataset(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")
        
        def copy_file(src_dest):
            src, dest = src_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        
        source_files = list(self.source_path.rglob("*"))
        copy_pairs = [
            (src, self.temp_path / src.relative_to(self.source_path))
            for src in source_files if src.is_file()
        ]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(tqdm(
                executor.map(copy_file, copy_pairs),
                total=len(copy_pairs),
                desc="Copying dataset"
            ))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = EnglishHandwrittenProcessor()
    processor.process()
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

            self._parallel_copy_dataset()
            self._simplify_documentation()
            
            image_files = list((self.temp_path / "images").glob("*.png"))
            chunks = [image_files[i:i + self.chunk_size] for i in range(0, len(image_files), self.chunk_size)]
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                list(tqdm(
                    executor.map(self._process_chunk, chunks),
                    total=len(chunks),
                    desc="Processing image chunks"
                ))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}")
            return False
            
    def _process_chunk(self, image_paths: List[Path]) -> None:
        try:
            for img_path in image_paths:
                try:
                    cache_path = self.cache_dir / img_path.name
                    if cache_path.exists():
                        shutil.copy2(cache_path, img_path)
                        continue
                    
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Only resize and invert
                        processed = self._process_single_image(image)
                        if processed is not None:
                            cv2.imwrite(str(cache_path), processed)
                            cv2.imwrite(str(img_path), processed)
                except Exception as e:
                    self.logger.warning(f"Error processing {img_path}: {str(e)}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error in chunk processing: {str(e)}")

    def _process_single_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            # Resize to target size
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            # Invert colors
            return 255 - image
            
        except Exception as e:
            self.logger.warning(f"Error in single image processing: {str(e)}")
            return None

    def _simplify_documentation(self) -> None:
        try:
            doc_path = self.temp_path / "documentation.txt"
            if not doc_path.exists():
                return

            simplified_lines = []
            with open(doc_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        simplified_lines.append(f"{parts[0]} {parts[-1]}\n")

            with open(doc_path, 'w') as f:
                f.writelines(simplified_lines)

        except Exception as e:
            self.logger.error(f"Error simplifying documentation: {str(e)}")

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
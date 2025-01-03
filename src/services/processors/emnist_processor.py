import logging
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class EMNISTProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "emnist"
        self.temp_path = self.project_root / "data" / "temp" / "EMNIST"
        
    def _copy_dataset(self) -> None:
        if self.source_path.exists():
            shutil.copytree(self.source_path, self.temp_path, dirs_exist_ok=True)
            self.logger.info(f"Copied dataset to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")

    def _enhance_white_line(self, image: np.ndarray) -> np.ndarray:
        _, white_mask = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2,2), np.uint8)
        enhanced = cv2.dilate(white_mask, kernel, iterations=1)
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0.5)
        return enhanced

    def process(self) -> bool:
        try:
            self.logger.info("Processing EMNIST dataset...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
                
            self._copy_dataset()
            image_files = list((self.temp_path / "images").glob("*.png"))
            
            for img_path in tqdm(image_files, desc="Enhancing white lines"):
                try:
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue
                    enhanced = self._enhance_white_line(image)
                    cv2.imwrite(str(img_path), enhanced)
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            self.logger.info("EMNIST processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in EMNIST processing: {str(e)}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = EMNISTProcessor()
    processor.process()
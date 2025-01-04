import logging
import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm

class Chars74KProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "chars74k"
        self.temp_path = self.project_root / "data" / "temp" / "Chars74K"

    def _normalize_documentation(self) -> None:
        """Normalize documentation to simplified format (filename label)."""
        doc_path = self.temp_path / "documentation.txt"
        normalized_path = self.temp_path / "normalized_documentation.txt"
        
        skip_lines = 3  # Number of header lines to skip
        
        with open(doc_path, 'r') as f, open(normalized_path, 'w') as out:
            # Skip header lines
            for _ in range(skip_lines):
                next(f, None)
                
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]        # First column (filename)
                    label = parts[-1]          # Last column (label)
                    out.write(f"{filename} {label}\n")

        normalized_path.replace(doc_path)
        self.logger.info("Documentation normalized to simplified format")
        
    def process(self) -> bool:
        try:
            self.logger.info("Processing Chars74K dataset...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
                
            self._copy_dataset()
            self._normalize_documentation()
            image_files = list((self.temp_path / "images").glob("*.png"))
            total_files = len(image_files)
            
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing Chars74K")):
                try:
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {img_path}")
                        continue
                        
                    processed = self._process_single_image(image)
                    cv2.imwrite(str(img_path), processed)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            self.logger.info("Chars74K processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in Chars74K processing: {str(e)}")
            return False

    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process single Chars74K image by inverting colors."""
        return cv2.bitwise_not(image)

    def _copy_dataset(self) -> None:
        """Copy dataset to temp directory."""
        if self.source_path.exists():
            shutil.copytree(self.source_path, self.temp_path, dirs_exist_ok=True)
            self.logger.info(f"Copied dataset to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")

def test_processor():
    """Test Chars74K processor functionality."""
    import tempfile
    
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        img_dir = test_dir / "data" / "raw" / "chars74k" / "images"
        img_dir.mkdir(parents=True)
        
        # Create test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(test_image, "R", (25, 75), cv2.FONT_HERSHEY_COMPLEX, 2, 255, 2)
        cv2.imwrite(str(img_dir / "test.png"), test_image)
        
        processor = Chars74KProcessor()
        processor.project_root = test_dir
        processor.source_path = test_dir / "data" / "raw" / "chars74k"
        processor.temp_path = test_dir / "data" / "temp" / "Chars74K"
        
        assert processor.process() == True
        
        processed_path = processor.temp_path / "images" / "test.png"
        assert processed_path.exists()
        
        processed_img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        assert processed_img is not None
        
        print("All tests passed!")
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_processor()
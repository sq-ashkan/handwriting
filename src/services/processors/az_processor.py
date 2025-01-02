import logging
import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm

class AZProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "az_handwritten"
        self.temp_path = self.project_root / "data" / "temp" / "AZ"

        

    def _normalize_documentation(self) -> None:
        doc_path = self.temp_path / "documentation.txt"
        normalized_path = self.temp_path / "normalized_documentation.txt"
        
        with open(doc_path, 'r') as f, open(normalized_path, 'w') as out:
            out.write("# IAM Format: filename label grayscale components x y width height tag transcription\n\n")
            
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 10:
                    filename = parts[0]
                    label = parts[-1]
                    
                    normalized_filename = f"EH_{filename.replace('-', '_')}"
                    normalized_line = f"{normalized_filename} 1 255 1 0 0 27 27 AZ {label}\n"
                    out.write(normalized_line)

        normalized_path.replace(doc_path)
        self.logger.info("Documentation normalized to IAM format")    
        
    def process(self) -> bool:
        try:
            self.logger.info("Processing A-Z Handwritten dataset...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
                
            self._copy_dataset()
            self._normalize_documentation()
            
            image_files = list((self.temp_path / "images").glob("*.png"))
            total_files = len(image_files)
            
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing A-Z")):
                try:
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {img_path}")
                        continue
                        
                    processed = self._process_single_image(image)
                    cv2.imwrite(str(img_path), processed)
                    
                    progress = (idx + 1) / total_files * 100
                    self.logger.debug(f"Processed {img_path.name} ({progress:.2f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            self.logger.info("A-Z processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in A-Z processing: {str(e)}")
            return False

    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process single A-Z image through enhancement pipeline."""
        image = self._enhance_contrast(image)
        image = self._reduce_noise(image)
        image = self._center_character(image)
        image = self._normalize_stroke(image)
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise preserving edges."""
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _center_character(self, image: np.ndarray) -> np.ndarray:
        """Center character using moments."""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
            
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return image
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        rows, cols = image.shape
        shift_x = (cols // 2) - cx
        shift_y = (rows // 2) - cy
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, (cols, rows))

    def _normalize_stroke(self, image: np.ndarray) -> np.ndarray:
        """Normalize stroke width."""
        kernel_size = max(2, min(image.shape) // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def _copy_dataset(self) -> None:
        """Copy dataset to temp directory."""
        if self.source_path.exists():
            shutil.copytree(self.source_path, self.temp_path, dirs_exist_ok=True)
            self.logger.info(f"Copied dataset to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")

def test_processor():
    """Test AZ processor functionality."""
    import tempfile
    
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        img_dir = test_dir / "data" / "raw" / "az_handwritten" / "images"
        img_dir.mkdir(parents=True)
        
        # Create test image
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(test_image, "A", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 2)
        
        # Add noise
        noise = np.random.normal(0, 25, test_image.shape).astype(np.uint8)
        noisy_image = cv2.add(test_image, noise)
        
        cv2.imwrite(str(img_dir / "test.png"), noisy_image)
        
        processor = AZProcessor()
        processor.project_root = test_dir
        processor.source_path = test_dir / "data" / "raw" / "az_handwritten"
        processor.temp_path = test_dir / "data" / "temp" / "AZ"
        
        assert processor.process() == True
        
        processed_path = processor.temp_path / "images" / "test.png"
        assert processed_path.exists()
        
        processed_img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        assert processed_img is not None
        assert np.mean(processed_img) > 0
        assert len(np.unique(processed_img)) > 1
        
        print("All tests passed!")
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_processor()
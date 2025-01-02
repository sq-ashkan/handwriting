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
        doc_path = self.temp_path / "documentation.txt"
        
        with open(doc_path, 'w') as out:
            out.write("# IAM Format: filename label grayscale components x y width height tag transcription\n\n")
            
            image_files = list((self.temp_path / "images").glob("*.png"))
            for img_path in image_files:
                filename = img_path.stem
                label = filename.split('_')[0]  # استخراج لیبل از نام فایل
                normalized_line = f"EH_{filename} 1 255 1 0 0 27 27 CHARS74K {label}\n"
                out.write(normalized_line)

        self.logger.info("Documentation created in IAM format")
        
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
                    
                    progress = (idx + 1) / total_files * 100
                    self.logger.debug(f"Processed {img_path.name} ({progress:.2f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            self.logger.info("Chars74K processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in Chars74K processing: {str(e)}")
            return False

    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process single Chars74K image through enhancement pipeline."""
        # Normalize brightness
        image = self._normalize_brightness(image)
        
        # Reduce noise
        image = self._reduce_noise(image)
        
        # Center character
        image = self._center_character(image)
        
        # Normalize stroke width
        image = self._normalize_stroke(image)
        
        return image

    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness using adaptive histogram equalization."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving character edges."""
        # Non-local means denoising for better edge preservation
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

    def _center_character(self, image: np.ndarray) -> np.ndarray:
        """Center character using connected component analysis."""
        # Find all connected components
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
        
        if len(stats) < 2:  # No character found
            return image
            
        # Get largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Get component centroid
        cy, cx = map(int, centroids[largest_label])
        
        # Calculate shift needed
        rows, cols = image.shape
        shift_x = (cols // 2) - cx
        shift_y = (rows // 2) - cy
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        return cv2.warpAffine(image, M, (cols, rows))

    def _normalize_stroke(self, image: np.ndarray) -> np.ndarray:
        """Normalize stroke width using morphological operations."""
        # Estimate stroke width using distance transform
        dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        avg_stroke = int(np.mean(dist[dist > 0]))
        
        # Create kernel based on estimated stroke width
        kernel_size = max(2, avg_stroke // 2)
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
    """Test Chars74K processor functionality."""
    import tempfile
    
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        img_dir = test_dir / "data" / "raw" / "chars74k" / "images"
        img_dir.mkdir(parents=True)
        
        # Create test image with varied stroke width
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(test_image, "R", (25, 75), cv2.FONT_HERSHEY_COMPLEX, 2, 255, 3)
        cv2.putText(test_image, "R", (26, 76), cv2.FONT_HERSHEY_COMPLEX, 2, 255, 1)
        
        # Add varying brightness and noise
        noise = np.random.normal(0, 30, test_image.shape).astype(np.uint8)
        gradient = np.linspace(0, 50, 100).reshape(-1, 1)
        noisy_image = cv2.add(test_image, noise)
        noisy_image = cv2.add(noisy_image, gradient)
        
        cv2.imwrite(str(img_dir / "test.png"), noisy_image)
        
        processor = Chars74KProcessor()
        processor.project_root = test_dir
        processor.source_path = test_dir / "data" / "raw" / "chars74k"
        processor.temp_path = test_dir / "data" / "temp" / "Chars74K"
        
        assert processor.process() == True
        
        processed_path = processor.temp_path / "images" / "test.png"
        assert processed_path.exists()
        
        processed_img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        assert processed_img is not None
        
        # Check brightness normalization
        std_dev = np.std(processed_img)
        assert std_dev > 0, "Image has no contrast"
        
        # Check noise reduction
        unique_values = len(np.unique(processed_img))
        assert unique_values < len(np.unique(noisy_image)), "Noise not reduced"
        
        # Check if character is centered
        moments = cv2.moments(processed_img)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            assert abs(cx - processed_img.shape[1]/2) < 10, "Character not centered horizontally"
            assert abs(cy - processed_img.shape[0]/2) < 10, "Character not centered vertically"
        
        print("All tests passed!")
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_processor()
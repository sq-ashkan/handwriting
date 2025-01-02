import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

class EMNISTProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "emnist"
        self.temp_path = self.project_root / "data" / "temp" / "EMNIST"
    

    def _normalize_documentation(self) -> None:
        doc_path = self.temp_path / "documentation.txt"
        normalized_path = self.temp_path / "normalized_documentation.txt"
        
        with open(doc_path, 'r') as f, open(normalized_path, 'w') as out:
            out.write("# IAM Format: filename label grayscale components x y width height tag transcription\n\n")
            
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split()
                filename = parts[0]
                normalized_filename = f"EH_{filename}"
                normalized_line = f"{normalized_filename} 1 255 1 0 0 28 28 EMNIST {parts[-1]}\n"
                out.write(normalized_line)
        
        normalized_path.replace(doc_path)
        
    def process(self) -> bool:
        try:
            self.logger.info("Processing EMNIST dataset...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
                
            self._copy_dataset()
            self._normalize_documentation() 
            
            image_files = list((self.temp_path / "images").glob("*.png"))
            total_files = len(image_files)
            
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing EMNIST")):
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
                    
            self.logger.info("EMNIST processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in EMNIST processing: {str(e)}")
            return False

    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single EMNIST image through enhancement pipeline."""
        # Convert to float32 for better precision
        image = image.astype(np.float32) / 255.0
        
        # Enhance contrast using adaptive histogram equalization
        image = self._enhance_contrast(image)
        
        # Denoise while preserving edges
        image = self._reduce_noise(image)
        
        # Center the character
        image = self._center_character(image)
        
        # Normalize stroke width
        image = self._normalize_stroke(image)
        
        # Convert back to uint8
        return (image * 255).astype(np.uint8)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE with optimal parameters."""
        image_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(image_uint8)
        return enhanced.astype(np.float32) / 255.0

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise while preserving character edges."""
        # Convert to uint8 for OpenCV operations
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Apply bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(image_uint8, d=5, sigmaColor=50, sigmaSpace=50)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary.astype(np.float32) / 255.0

    def _center_character(self, image: np.ndarray) -> np.ndarray:
        """Center the character using precise moment calculation."""
        # Convert to binary for contour detection
        binary = (image > 0.5).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return image
            
        # Calculate center with sub-pixel precision
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        rows, cols = image.shape
        shift_x = (cols / 2.0) - cx
        shift_y = (rows / 2.0) - cy
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

    def _normalize_stroke(self, image: np.ndarray) -> np.ndarray:
        """Normalize stroke width using adaptive morphological operations."""
        binary = (image > 0.5).astype(np.uint8) * 255
        
        # Calculate average stroke width using distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        avg_stroke = np.mean(dist[dist > 0]) * 2
        
        # Adjust kernel size based on stroke width
        kernel_size = max(2, int(avg_stroke + 0.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological operations
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        normalized = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return normalized.astype(np.float32) / 255.0

    def _copy_dataset(self) -> None:
        """Copy dataset files to temp directory."""
        if self.source_path.exists():
            shutil.copytree(self.source_path, self.temp_path, dirs_exist_ok=True)
            self.logger.info(f"Copied dataset to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")

def test_processor():
    """Test EMNIST processor with comprehensive verification."""
    import tempfile
    
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        img_dir = test_dir / "data" / "raw" / "emnist" / "images"
        img_dir.mkdir(parents=True)
        
        # Create test patterns
        test_images = []
        
        # Pattern 1: Simple rectangle
        img1 = np.zeros((28, 28), dtype=np.uint8)
        cv2.rectangle(img1, (8, 8), (20, 20), 255, -1)
        test_images.append(("test1.png", img1))
        
        # Pattern 2: Thin stroke character
        img2 = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(img2, "E", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1)
        test_images.append(("test2.png", img2))
        
        # Add noise and save test images
        for filename, img in test_images:
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            noisy_image = cv2.add(img, noise)
            cv2.imwrite(str(img_dir / filename), noisy_image)
        
        # Initialize processor
        processor = EMNISTProcessor()
        processor.project_root = test_dir
        processor.source_path = test_dir / "data" / "raw" / "emnist"
        processor.temp_path = test_dir / "data" / "temp" / "EMNIST"
        
        assert processor.process() == True
        
        # Verify processed images
        for filename, _ in test_images:
            processed_path = processor.temp_path / "images" / filename
            assert processed_path.exists()
            
            processed_img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
            assert processed_img is not None
            assert processed_img.shape == (28, 28)
            
            # Check contrast
            hist = cv2.calcHist([processed_img], [0], None, [256], [0,256])
            assert np.max(hist) > np.min(hist) * 3
            
            # Check noise reduction
            assert len(np.unique(processed_img)) < len(np.unique(noisy_image))
            
            # Check centering
            moments = cv2.moments(processed_img)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                assert abs(cx - processed_img.shape[1]/2) < 2
                assert abs(cy - processed_img.shape[0]/2) < 2
        
        print("All tests passed successfully!")
        
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_processor()
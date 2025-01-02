import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np
from tqdm import tqdm

class EnglishHandwrittenProcessor:
    """Processor for English handwritten character images."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "english_handwritten"
        self.temp_path = self.project_root / "data" / "temp" / "EH"
        self.target_size = (27, 27)
        
    def process(self) -> bool:
        """Main processing pipeline for the English handwritten dataset."""
        try:
            self.logger.info("Starting English handwritten text processing...")
            
            # Create temp directory if it doesn't exist
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if not self.source_path.exists():
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
                
            # Copy dataset to temp location
            self._copy_dataset()
            self.logger.info(f"Dataset copied to {self.temp_path}")
            
            # Process all images
            image_files = list((self.temp_path / "images").glob("*.png"))
            total_files = len(image_files)
            
            for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
                try:
                    # Read image
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {img_path}")
                        continue
                        
                    # Process image
                    processed = self._process_single_image(image)
                    
                    # Save processed image
                    cv2.imwrite(str(img_path), processed)
                    
                    # Log progress
                    progress = (idx + 1) / total_files * 100
                    self.logger.info(f"Processed {img_path.name} ({progress:.2f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_path}: {str(e)}")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error in English handwritten processing: {str(e)}")
            return False
            
    def _process_single_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single character image through all steps.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Processed image
        """
        # Remove noise and normalize brightness
        image = self._normalize_brightness(image)
        
        # Remove margins and center character
        image = self._remove_margins(image)
        
        # Center the character
        image = self._center_character(image)
        
        # Normalize stroke width
        image = self._normalize_stroke_width(image)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        return image
        
    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness and remove noise.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Normalized image
        """
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
        
    def _remove_margins(self, image: np.ndarray) -> np.ndarray:
        """Remove excess margins around the character.
        
        Args:
            image: Binary image
            
        Returns:
            Image with margins removed
        """
        # Find contours
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        # Crop image
        return image[y:y+h, x:x+w]
        
    def _center_character(self, image: np.ndarray) -> np.ndarray:
        """Center the character in the image.
        
        Args:
            image: Binary image
            
        Returns:
            Centered image
        """
        # Get image dimensions
        h, w = image.shape
        size = max(h, w) + 10  # Add padding
        
        # Create square background
        background = np.zeros((size, size), dtype=np.uint8)
        
        # Calculate position to paste
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        
        # Paste original image
        background[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        return background
        
    def _normalize_stroke_width(self, image: np.ndarray) -> np.ndarray:
        """Normalize the stroke width of the character.
        
        Args:
            image: Binary image
            
        Returns:
            Image with normalized stroke width
        """
        # Apply slight dilation to ensure consistent stroke width
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        return dilated
        
    def _copy_dataset(self) -> None:
        """Copy dataset files to temp directory."""
        if self.source_path.exists():
            # Copy entire directory tree
            shutil.copytree(
                self.source_path, 
                self.temp_path, 
                dirs_exist_ok=True
            )
            self.logger.info(f"Copied dataset to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")
            
# Tests
def test_processor():
    """Test the EnglishHandwrittenProcessor class."""
    import pytest
    import tempfile
    import shutil
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test image
        img_dir = test_dir / "data" / "raw" / "english_handwritten" / "images"
        img_dir.mkdir(parents=True)
        
        # Create test image (white background with black character)
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (40, 30), (60, 70), 0, -1)  # Draw black rectangle
        
        # Save test image
        cv2.imwrite(str(img_dir / "test.png"), test_image)
        
        # Initialize processor
        processor = EnglishHandwrittenProcessor()
        processor.project_root = test_dir
        processor.source_path = test_dir / "data" / "raw" / "english_handwritten"
        processor.temp_path = test_dir / "data" / "temp" / "EH"
        
        # Run processing
        assert processor.process() == True
        
        # Check if processed image exists
        processed_path = processor.temp_path / "images" / "test.png"
        assert processed_path.exists()
        
        # Load processed image and verify size
        processed_img = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
        assert processed_img.shape == (27, 27)
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    test_processor()
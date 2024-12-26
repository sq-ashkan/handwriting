# src/services/preprocessor.py
import cv2
import numpy as np
from pathlib import Path
from src.lib.utils import setup_logging

class DatasetPreprocessor:
    def __init__(self):
        self.logger = setup_logging("preprocessor")
        
    def normalize_size(self, image, target_size=(128, 128)):
        """نرمال‌سازی اندازه تصویر"""
        return cv2.resize(image, target_size)
        
    def convert_to_grayscale(self, image):
        """تبدیل به سیاه و سفید"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    def remove_noise(self, image):
        """حذف نویز با فیلتر گوسی"""
        return cv2.GaussianBlur(image, (5,5), 0)
        
    def enhance_contrast(self, image):
        """افزایش کنتراست"""
        return cv2.equalizeHist(image)
        
    def correct_skew(self, image):
        """تصحیح چرخش متن"""
        # الگوریتم تشخیص و تصحیح زاویه
        pass
        
    def segment_characters(self, image):
        """جداسازی کاراکترها"""
        # الگوریتم segmentation
        pass
        
    def process_image(self, image_path: Path):
        """پردازش کامل یک تصویر"""
        try:
            # خواندن تصویر
            image = cv2.imread(str(image_path))
            
            # اعمال پردازش‌ها
            image = self.normalize_size(image)
            image = self.convert_to_grayscale(image)
            image = self.remove_noise(image)
            image = self.enhance_contrast(image)
            
            return image
            
        except Exception as e:
            self.logger.error(f"خطا در پردازش تصویر {image_path}: {str(e)}")
            return None
            
    def process_dataset(self, input_dir: Path, output_dir: Path):
        """پردازش کل دیتاست"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for image_path in input_dir.glob('*.png'):  # یا هر فرمت دیگری
                processed_image = self.process_image(image_path)
                if processed_image is not None:
                    output_path = output_dir / image_path.name
                    cv2.imwrite(str(output_path), processed_image)
                    
            self.logger.info("پردازش دیتاست با موفقیت انجام شد")
            return True
            
        except Exception as e:
            self.logger.error(f"خطا در پردازش دیتاست: {str(e)}")
            return False
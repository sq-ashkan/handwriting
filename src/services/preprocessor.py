import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl
from concurrent.futures import ThreadPoolExecutor
import logging
from src.lib.utils import setup_logging
from src.lib.config import Config

class DatasetPreprocessor:
    def __init__(self, config: Config):
        """
        # راه‌اندازی پردازشگر داده با تنظیمات ورودی
        # تنظیم لاگر و پیکربندی اولیه پردازشگر تصویر
        """
        self.config = config
        self.logger = setup_logging(__name__)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
    def normalize_size(self, image: np.ndarray) -> np.ndarray:
        """
        # تغییر اندازه تصویر به سایز استاندارد تعریف شده در تنظیمات
        # تضمین می‌کند همه تصاویر اندازه یکسان داشته باشند
        """
        return cv2.resize(image, self.config.IMAGE_SIZE)
        
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        # تبدیل تصویر رنگی به سیاه و سفید
        # برای کاهش پیچیدگی و تمرکز روی محتوای نوشته
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
        
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        # تبدیل تصویر به حالت دودویی با استفاده از روش آستانه‌گذاری اتسو
        # بهبود تمایز بین نوشته و پس‌زمینه
        """
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
        
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        # حذف نویز تصویر با استفاده از فیلتر گوسی
        # کاهش اختلالات تصویر برای تشخیص بهتر
        """
        return cv2.GaussianBlur(image, (3, 3), 0)
        
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        # بهبود کنتراست تصویر با استفاده از هیستوگرام متعادل
        # افزایش وضوح بین نوشته و پس‌زمینه
        """
        return cv2.equalizeHist(image)
        
    def add_padding(self, image: np.ndarray) -> np.ndarray:
        """
        # اضافه کردن حاشیه به تصویر
        # برای اطمینان از عدم برش نوشته‌ها در لبه‌ها
        """
        top = bottom = left = right = self.config.PADDING
        return cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=255)
                                
    def augment_data(self, image: np.ndarray) -> List[np.ndarray]:
        """
        # افزایش داده‌ها با روش‌های مختلف
        # برای بهبود یادگیری مدل و افزایش مقاومت
        """
        augmented_images = []
        
        # چرخش تصادفی با زاویه محدود
        angle = np.random.uniform(self.config.ROTATION_RANGE[0],
                                self.config.ROTATION_RANGE[1])
        rows, cols = image.shape
        matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, matrix, (cols, rows))
        augmented_images.append(rotated)
        
        # تغییر مقیاس تصادفی
        scale = np.random.uniform(self.config.SCALE_RANGE[0],
                                self.config.SCALE_RANGE[1])
        scaled = cv2.resize(image, None, fx=scale, fy=scale)
        scaled = cv2.resize(scaled, (cols, rows))
        augmented_images.append(scaled)
        
        # افزودن نویز گوسی
        noise = np.random.normal(0, self.config.NOISE_VARIANCE, image.shape)
        noisy = np.clip(image + noise * 255, 0, 255).astype(np.uint8)
        augmented_images.append(noisy)
        
        return augmented_images
        
    def process_single_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """
        # پردازش کامل یک تصویر شامل تمام مراحل نرمال‌سازی و بهبود
        # تبدیل نهایی به تنسور پایتورچ
        """
        try:
            self.logger.info(f"Processing image: {image_path.name}")
            
            # خواندن تصویر
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None
                
            # اعمال پردازش‌های پایه
            image = self.convert_to_grayscale(image)
            image = self.normalize_size(image)
            image = self.remove_noise(image)
            image = self.enhance_contrast(image)
            image = self.binarize_image(image)
            image = self.add_padding(image)
            
            # تبدیل به تنسور
            tensor_image = self.transform(image)
            
            self.logger.info(f"Successfully processed: {image_path.name}")
            return tensor_image
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path.name}: {str(e)}")
            return None
            
    def process_dataset(self, input_dir: Path, output_dir: Path) -> bool:
        """
        # پردازش موازی کل مجموعه داده با استفاده از چند نخ
        # ذخیره نتایج در مسیر خروجی
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Starting dataset processing from {input_dir}")
            
            with ThreadPoolExecutor(max_workers=self.config.NUM_WORKERS) as executor:
                futures = []
                for image_path in input_dir.glob('*.png'):
                    future = executor.submit(self.process_single_image, image_path)
                    futures.append((future, image_path))
                    
                for future, image_path in futures:
                    try:
                        processed_tensor = future.result()
                        if processed_tensor is not None:
                            output_path = output_dir / f"{image_path.stem}_processed{image_path.suffix}"
                            # تبدیل تنسور به تصویر و ذخیره
                            processed_image = (processed_tensor.numpy() * 255).astype(np.uint8)
                            cv2.imwrite(str(output_path), processed_image)
                    except Exception as e:
                        self.logger.error(f"Error saving {image_path.name}: {str(e)}")
                        
            self.logger.info("Dataset processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset processing failed: {str(e)}")
            return False
            
class IAMDataModule(pl.LightningDataModule):
    """
    # ماژول داده پایتورچ لایتنینگ برای مجموعه داده
    # مدیریت بارگذاری و پردازش داده‌ها
    """
    def __init__(self, data_dir: str, config: Config):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.preprocessor = DatasetPreprocessor(config)
        
    def prepare_data(self):
        """
        # آماده‌سازی داده‌ها قبل از شروع آموزش
        # شامل دانلود و پردازش اولیه
        """
        raw_dir = self.data_dir / "raw"
        processed_dir = self.data_dir / "processed"
        self.preprocessor.process_dataset(raw_dir, processed_dir)
        
    def setup(self, stage: Optional[str] = None):
        """
        # تنظیم مجموعه‌های داده برای آموزش و ارزیابی
        # تقسیم داده‌ها به بخش‌های مختلف
        """
        if stage == "fit" or stage is None:
            # تنظیم داده‌های آموزش و اعتبارسنجی
            pass
            
        if stage == "test" or stage is None:
            # تنظیم داده‌های آزمون
            pass
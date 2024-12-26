# برای دسترسی و استفاده از امکانات سیستم عامل
import os
import logging
from pathlib import Path
# برای دریافت مجموعه داده از پایگاه داده
import kagglehub # type: ignore
# برای جابجایی و مدیریت پیشرفته پرونده‌ها
import shutil

from src.lib.utils import create_directories
from src.lib.constants import PROCESSED_DIR

class DatasetDownloader:
    def download_dataset(self) -> Path:
        # تابعی برای دریافت مجموعه داده از سایت کگل
        # خروجی تابع از نوع Path است که مسیر دانلود را مشخص میکند
        
        # ثبت شروع فرایند دانلود در گزارش‌ها
        logging.info("Starting dataset download from Kaggle")
        
        try:
            # تلاش برای دانلود مجموعه داده با استفاده از کتابخانه کگل
            # مجموعه داده دست‌نوشته از کاربر نادر عبدالغنی
            dataset_path = kagglehub.dataset_download(
                "naderabdalghani/iam-handwritten-forms-dataset"
            )
            
            # تبدیل مسیر دریافتی به شیء از نوع Path برای مدیریت بهتر مسیر
            dataset_path = Path(dataset_path)
            
            # ثبت موفقیت‌آمیز بودن دانلود در گزارش‌ها
            logging.info(f"Dataset downloaded successfully to: {dataset_path}")
            
            # برگرداندن مسیر دانلود
            return dataset_path
            
        except Exception as e:
            # در صورت بروز هر نوع خطا:
            # ۱- ثبت خطا در گزارش‌ها
            logging.error(f"Failed to download dataset: {str(e)}")
            # ۲- انتقال خطا به سطح بالاتر
            raise

    def process_dataset(self, dataset_path: Path) -> bool:
        # تابعی برای پردازش و سازماندهی پرونده‌های دانلود شده
        # ورودی: مسیر پوشه دانلود شده
        # خروجی: درستی یا نادرستی عملیات
        
        try:
            # ثبت شروع پردازش پرونده‌ها
            logging.info("Processing downloaded files")
            
            # شروع عملیات رونوشت‌برداری به پوشه پردازش شده
            logging.info(f"Copying files to: {PROCESSED_DIR}")
            
            # بررسی همه موارد در مسیر دانلود
            for item in dataset_path.glob('*'):
                # اگر مورد یک پرونده باشد
                if item.is_file():
                    # رونوشت‌برداری پرونده با حفظ ویژگی‌ها
                    shutil.copy2(item, PROCESSED_DIR)
                
                # اگر مورد یک پوشه باشد    
                elif item.is_dir():
                    # رونوشت‌برداری کل پوشه با همه محتویات
                    # dirs_exist_ok=True: اگر پوشه وجود داشت خطا ندهد
                    shutil.copytree(item, PROCESSED_DIR / item.name, dirs_exist_ok=True)
            
            # ثبت موفقیت‌آمیز بودن عملیات
            logging.info("Files processed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to process files: {str(e)}")
            return False

    def verify_dataset(self) -> bool:
        # تابعی برای اعتبارسنجی داده‌های دریافت و پردازش شده
        # خروجی: درستی یا نادرستی اعتبارسنجی
        
        try:
            # ثبت شروع فرایند اعتبارسنجی
            logging.info("Verifying downloaded dataset")
            
            # فهرست پرونده‌ها و پوشه‌های ضروری
            # در اینجا فقط پوشه 'data' بررسی میشود
            required_files = ['data']
            
            # بررسی وجود هر پرونده یا پوشه ضروری
            for file in required_files:
                # ساخت مسیر کامل برای بررسی
                file_path = PROCESSED_DIR / file
                
                # اگر پرونده یا پوشه وجود نداشت
                if not file_path.exists():
                    # ثبت خطا در گزارش‌ها
                    logging.error(f"Missing required file/directory: {file}")
                    # برگرداندن نتیجه نادرست
                    return False
            
            # ثبت موفقیت‌آمیز بودن اعتبارسنجی
            logging.info("Dataset verification completed successfully")
            return True
            
        # در صورت بروز هر خطای پیش‌بینی نشده
        except Exception as e:
            # ثبت خطا در گزارش‌ها
            logging.error(f"Dataset verification failed: {str(e)}")
            # برگرداندن نتیجه نادرست
            return False

    def run(self) -> bool:
        try:
            # ساخت پوشه‌های مورد نیاز
            create_directories()
            
            # مرحله ۲: دریافت داده‌ها
            dataset_path = self.download_dataset()
            
            # مرحله ۳: پردازش و ذخیره‌سازی
            # اگر پردازش با شکست مواجه شد
            if not self.process_dataset(dataset_path):
                return False
                
            # مرحله ۴: بررسی نهایی
            # اگر بررسی با شکست مواجه شد
            if not self.verify_dataset():
                return False
                
            # ثبت موفقیت‌آمیز بودن کل فرایند
            logging.info("Dataset preparation completed successfully")
            return True
            
        # در صورت بروز هر خطای پیش‌بینی نشده
        except Exception as e:
            logging.error(f"Dataset preparation failed: {str(e)}")
            return False
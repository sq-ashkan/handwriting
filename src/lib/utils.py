# برای کنترل متغیرهای محیطی و خروج کنترل شده از برنامه
import sys
# برای نگهداری سوابق و ثبت رخدادهای برنامه
import logging
from pathlib import Path
from .constants import LOGS_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR

def setup_logging():
    # تعریف تابعی برای راه‌اندازی سامانه ثبت رویدادها
    
    # تعیین مسیر و نام پرونده برای ذخیره رویدادها
    log_file = LOGS_DIR / "download.log"
    
    # ایجاد پوشه برای ذخیره رویدادها - اگر وجود نداشت میسازد
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # تنظیمات پایه برای ثبت رویدادها
    logging.basicConfig(
        # تعیین سطح اهمیت رویدادها - در اینجا همه پیام‌های مهم ثبت میشوند
        level=logging.INFO,
        
        # قالب‌بندی پیام‌های ثبت شده:
        # زمان - سطح اهمیت - متن پیام
        format='%(asctime)s - %(levelname)s - %(message)s',
        
        # تعیین مکان‌های ذخیره رویدادها:
        handlers=[
            # ذخیره در پرونده تعیین شده
            logging.FileHandler(log_file),
            
            # نمایش همزمان در خروجی برنامه
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    # تابعی برای ساخت پوشه‌های مورد نیاز در پروژه
    
    # حلقه روی تمام مسیرهای تعریف شده در بخش تنظیمات
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        
        # ساخت هر پوشه با دو ویژگی:
        # parents=True: اگر پوشه‌های والد وجود نداشته باشند، آنها را هم میسازد
        # exist_ok=True: اگر پوشه از قبل وجود داشته باشد، خطا نمیدهد
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # ثبت رویداد ساخت هر پوشه در سامانه ثبت رویدادها
        logging.info(f"Created directory: {dir_path}")
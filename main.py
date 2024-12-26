# برای کنترل متغیرهای محیطی و خروج کنترل شده از برنامه
import sys
from src.lib.utils import setup_logging
from src.services.downloader import DatasetDownloader
from src.lib.constants import PROCESSED_DIR

def main() -> bool:
    try:
        # راه‌اندازی سامانه ثبت رویدادها
        setup_logging()
        
        # ایجاد نمونه از کلاس دانلودکننده
        downloader = DatasetDownloader()
        
        # اجرای فرایند دانلود و دریافت نتیجه
        success = downloader.run()
        
        return success
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# نقطه شروع برنامه
if __name__ == "__main__":
    # اجرای تابع اصلی و دریافت نتیجه
    success = main()
    
    # نمایش محل ذخیره‌سازی داده‌ها
    print("\nDataset files location:", PROCESSED_DIR)
    
    # پایان برنامه با کد مناسب
    # اگر موفق بود: کد ۰
    # اگر ناموفق بود: کد ۱
    sys.exit(0 if success else 1)
# برای ساخت و مدیریت ساده‌تر مسیرهای پرونده‌ها
from pathlib import Path

# تنظیمات پایه
# تنظیمات مسیرهای اصلی پروژه
# مسیر ریشه پروژه - دو سطح بالاتر از فایل فعلی
PROJECT_ROOT = Path(__file__).parent.parent.parent

# مسیر پوشه داده‌ها در ریشه پروژه
DATA_DIR = PROJECT_ROOT / "data"

# مسیر پوشه داده‌های خام و پردازش نشده
RAW_DIR = DATA_DIR / "raw"

# مسیر پوشه داده‌های پردازش شده
PROCESSED_DIR = DATA_DIR / "processed"

# مسیر پوشه برای ذخیره گزارش‌ها و رویدادها
LOGS_DIR = DATA_DIR / "logs"
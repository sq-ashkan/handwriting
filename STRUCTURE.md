# پروژه تشخیص کاراکتر دست‌نویس (OCR)

## هدف اصلی
ساخت سیستمی که بتواند کاراکتر (حرف یا عدد) دست‌نویس که کاربر روی کاغذ می‌نویسد را آپلود و با دقت بالای ۹۹ درصد برای نمایش به سرمایه‌گذار در حالت دمو و تست کاربر واقعی تشخیص دهد.

## مشخصات محیط توسعه
- سیستم عامل: macOS
- پردازنده: Apple M2 Ultra
- اولویت: دقت بالا در شرایط واقعی بدون محدودیت زمانی و هزینه‌ای
- زبان کدنویسی: انگلیسی
- زبان مستندات: فارسی

## ساختار به‌روز شده پروژه
```
/Users/roammer/Documents/Github/handwriting/
├── src/                                # کدهای اصلی پروژه
│   ├── services/                       # سرویس‌های اصلی
│   │   ├── downloaders/                # دانلودرهای دیتاست
│   │   │   ├── english_handwritten.py  # دانلودر english_handwritten ✓
│   │   │   ├── emnist_downloader.py               # دانلودر EMNIST ✓
│   │   │   ├── mnist_downloader.py                # دانلودر MNIST ✓
│   │   │   ├── az_downloader.py                   # دانلودر A-Z ✓
│   │   │   └── chars74k_downloader.py             # دانلودر Chars74K ✓
│   │   ├── processors/                 # پردازش‌کننده‌ها
│   │   │   ├── english_handwritten.py
│   │   │   ├── emnist_processor.py
│   │   │   ├── mnist_processor.py
│   │   │   ├── az_processor.py
│   │   │   └── chars74k_processor.py
│   │   ├── preprocessor/     # پردازش تصاویر
│   │   └── trainer/          # سیستم آموزش
│   ├── models/               # معماری مدل
│   │   ├── layers.py         # لایه‌های شبکه
│   │   └── network.py        # پیکربندی شبکه
│   └── lib/                  # کتابخانه‌های کمکی
│       ├── utils.py          # توابع عمومی
│       ├── cache_manager.py  # مدیریت کش و فایل‌های موقت
│       └── constants.py      # متغیرهای ثابت
├── data/                     # دیتاست‌ها
│   ├── raw/                  # داده‌های خام
│   │   ├── english_handwritten/ # دیتاست english_handwritten
│   │   │   ├── images/         # تصاویر با فرمت یکسان
│   │   │   └── documentation.txt   # فرمت استاندارد
│   │   ├── emnist/             # دیتاست EMNIST
│   │   │   ├── images/         # تصاویر همسان با استاندارد
│   │   │   └── documentation.txt   # فرمت مشابه استاندارد
│   │   ├── mnist/              # دیتاست MNIST
│   │   │   ├── images/         # تصاویر همسان با استاندارد
│   │   │   └── documentation.txt   # فرمت مشابه استاندارد
│   │   ├── az_handwritten/    # دیتاست A-Z
│   │   │   ├── images/        # تصاویر همسان با استاندارد
│   │   │   └── documentation.txt  # فرمت مشابه استاندارد
│   │   └── chars74k/         # دیتاست Chars74K
│   │       ├── images/       # تصاویر ۲۷x۲۷ پیکسل
│   │       └── documentation.txt  # فرمت استاندارد
│   ├── processed/            # داده‌های پردازش شده
│   ├── temp/                 # داده‌های موقت
│   │   ├── EH/               # داده‌های موقت English Handwritten
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── EMNIST/           # داده‌های موقت EMNIST
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── MNIST/            # داده‌های موقت MNIST
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── AZ/               # داده‌های موقت A-Z
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   └── Chars74K/         # داده‌های موقت Chars74K
│   │       ├── images/    
│   │       └── documentation.txt
│   └── logs/                 # گزارش‌ها
├── tests/                    # تست‌ها
│   ├── test_downloaders/
│   ├── test_preprocessor/
│   └── test_trainer/
└── requirements.txt          # وابستگی‌ها
```

class Config:
    # تنظیمات پیش‌پردازش
    IMAGE_SIZE = (128, 32)  # برای IAM بهتره
    PADDING = 10
    MIN_TEXT_HEIGHT = 20
    BINARY_THRESHOLD = 128
    
    # تنظیمات افزایش داده
    ROTATION_RANGE = (-5, 5)
    SCALE_RANGE = (0.9, 1.1)
    NOISE_VARIANCE = 0.01
    
    # تنظیمات مدل
    MODEL_TYPE = "crnn"  # برای دست‌خط بهتر از ResNet عمل می‌کنه
    LEARNING_RATE = 0.0001  # کمتر برای ثبات بیشتر
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
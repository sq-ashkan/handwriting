import logging
from PIL import Image
import numpy as np

def debug_character(image_array, predicted_case, actual_case, char):
    """Debug helper to print image characteristics"""
    height, width = image_array.shape
    
    # Print basic image stats
    print(f"\nDebugging character: {char}")
    print(f"Predicted case: {'uppercase' if predicted_case else 'lowercase'}")
    print(f"Actual case: {actual_case}")
    print(f"Image shape: {image_array.shape}")
    
    # Print distribution of pixel values
    top_half = np.mean(image_array[:height//2, :])
    bottom_half = np.mean(image_array[height//2:, :])
    print(f"Top half average: {top_half:.3f}")
    print(f"Bottom half average: {bottom_half:.3f}")
    
    return top_half, bottom_half

# در این قسمت، کد اصلی دانلودر رو تغییر بدید تا این تابع رو صدا بزنه
# مثلا در متد process_csv_to_images:

def _process_csv_to_images(self, csv_path):
    records = []
    df = pd.read_csv(csv_path)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx > 10:  # فقط 10 تا تصویر اول رو چک کنیم
            break
            
        # تبدیل به تصویر
        pixel_data = row.values[1:].reshape(28, 28)
        
        # اینجا debug_character رو صدا بزنیم
        is_upper = self._determine_case(pixel_data, row.values[0])
        char = chr(row.values[0] + ord('a'))
        debug_character(pixel_data, is_upper, 'unknown', char)
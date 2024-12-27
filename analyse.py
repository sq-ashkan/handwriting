import os
import cv2
import numpy as np
from pathlib import Path

def analyze_dataset(dataset_path):
    """تحلیل یک دیتاست خاص و نمایش اطلاعات مهم آن"""
    
    results = {
        'name': os.path.basename(dataset_path),
        'doc_sample': None,
        'image_stats': {
            'count': 0,
            'sizes': set(),
            'formats': set(),
            'min_gray': float('inf'),
            'max_gray': -float('inf')
        }
    }
    
    # بررسی فایل documentation.txt
    doc_path = os.path.join(dataset_path, 'documentation.txt')
    if os.path.exists(doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            # نمونه 5 خط اول
            results['doc_sample'] = ''.join([next(f) for _ in range(5)])
    
    # بررسی تصاویر
    img_path = os.path.join(dataset_path, 'images')
    if os.path.exists(img_path):
        for img_file in os.listdir(img_path)[:100]:  # بررسی 100 تصویر اول
            img_full_path = os.path.join(img_path, img_file)
            try:
                img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    results['image_stats']['count'] += 1
                    results['image_stats']['sizes'].add(f"{img.shape[0]}x{img.shape[1]}")
                    results['image_stats']['formats'].add(Path(img_file).suffix)
                    results['image_stats']['min_gray'] = min(results['image_stats']['min_gray'], np.min(img))
                    results['image_stats']['max_gray'] = max(results['image_stats']['max_gray'], np.max(img))
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    return results

def main():
    datasets = [
        'data/raw/iam_handwriting',
        'data/raw/emnist',
        'data/raw/mnist',
        'data/raw/az_handwritten',
        'data/raw/chars74k'
    ]
    
    print("=== تحلیل دیتاست‌ها ===\n")
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            results = analyze_dataset(dataset_path)
            
            print(f"\n### دیتاست {results['name']} ###")
            print("\n[نمونه documentation.txt]:")
            print(results['doc_sample'] if results['doc_sample'] else "فایل documentation.txt یافت نشد")
            
            print("\n[آمار تصاویر]:")
            stats = results['image_stats']
            print(f"تعداد تصاویر بررسی شده: {stats['count']}")
            print(f"سایزهای مختلف: {', '.join(stats['sizes'])}")
            print(f"فرمت‌های موجود: {', '.join(stats['formats'])}")
            print(f"محدوده مقادیر خاکستری: {stats['min_gray']} تا {stats['max_gray']}")
            
            print("\n" + "="*50)
        else:
            print(f"\nمسیر {dataset_path} یافت نشد!")

if __name__ == "__main__":
    main()
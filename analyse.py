import os
import cv2
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

def extract_image_info_from_doc(doc_line):
    """Extract image information from a documentation line"""
    try:
        # Standard IAM pattern
        pattern = r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(.*)'
        match = re.match(pattern, doc_line)
        if match:
            return {
                'image_id': match.group(1),
                'status': match.group(2),
                'graylevel': int(match.group(3)),
                'components': int(match.group(4)),
                'x': int(match.group(5)),
                'y': int(match.group(6)),
                'width': int(match.group(7)),
                'height': int(match.group(8)),
                'tag': match.group(9),
                'transcription': match.group(10)
            }
    except Exception:
        pass
    return None

def analyze_documentation(doc_path):
    """Analyze documentation.txt file"""
    doc_stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'image_dimensions': defaultdict(int),
        'char_distribution': defaultdict(int),
        'sample_lines': []
    }
    
    if not os.path.exists(doc_path):
        return doc_stats
        
    with open(doc_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc_stats['total_lines'] += 1
            info = extract_image_info_from_doc(line)
            
            if info:
                doc_stats['valid_lines'] += 1
                dim = f"{info['width']}x{info['height']}"
                doc_stats['image_dimensions'][dim] += 1
                
                for char in info['transcription']:
                    if char.strip():  # Remove whitespace
                        doc_stats['char_distribution'][char] += 1
                        
            if doc_stats['total_lines'] <= 5:
                doc_stats['sample_lines'].append(line.strip())
                
    return doc_stats

def analyze_images(img_path, sample_size=1000):
    """Analyze dataset images"""
    img_stats = {
        'count': 0,
        'sizes': defaultdict(int),
        'formats': defaultdict(int),
        'gray_stats': {
            'min': float('inf'),
            'max': -float('inf'),
            'avg': 0,
            'std': 0
        },
        'aspect_ratios': defaultdict(int)
    }
    
    if not os.path.exists(img_path):
        return img_stats
        
    gray_values = []
    all_files = os.listdir(img_path)
    sample_files = all_files[:min(len(all_files), sample_size)]
    
    for img_file in sample_files:
        img_full_path = os.path.join(img_path, img_file)
        try:
            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_stats['count'] += 1
                
                # Size and aspect ratio
                h, w = img.shape
                size = f"{w}x{h}"
                img_stats['sizes'][size] += 1
                
                aspect = round(w/h, 2)
                img_stats['aspect_ratios'][aspect] += 1
                
                # File format
                fmt = Path(img_file).suffix
                img_stats['formats'][fmt] += 1
                
                # Grayscale statistics
                img_stats['gray_stats']['min'] = min(img_stats['gray_stats']['min'], np.min(img))
                img_stats['gray_stats']['max'] = max(img_stats['gray_stats']['max'], np.max(img))
                gray_values.extend(img.flatten())
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    if gray_values:
        img_stats['gray_stats']['avg'] = np.mean(gray_values)
        img_stats['gray_stats']['std'] = np.std(gray_values)
    
    return img_stats

def analyze_dataset(dataset_path):
    """Complete dataset analysis"""
    results = {
        'name': os.path.basename(dataset_path),
        'documentation': None,
        'images': None
    }
    
    # Analyze documentation
    doc_path = os.path.join(dataset_path, 'documentation.txt')
    results['documentation'] = analyze_documentation(doc_path)
    
    # Analyze images
    img_path = os.path.join(dataset_path, 'images')
    results['images'] = analyze_images(img_path)
    
    return results

def print_results(results):
    """Display analysis results"""
    print(f"\n=== Dataset Analysis: {results['name']} ===")
    
    # Documentation stats
    doc = results['documentation']
    print("\n[Documentation Statistics]:")
    print(f"Total lines: {doc['total_lines']}")
    print(f"Valid lines: {doc['valid_lines']}")
    print("\nImage dimensions in documentation:")
    for dim, count in sorted(doc['image_dimensions'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {dim}: {count} images")
    
    print("\nCharacter distribution (top 10):")
    for char, count in sorted(doc['char_distribution'].items(), key=lambda x: -x[1])[:10]:
        print(f"  '{char}': {count}")
    
    # Image stats
    img = results['images']
    print("\n[Image Statistics]:")
    print(f"Number of analyzed images: {img['count']}")
    
    print("\nCommon sizes:")
    for size, count in sorted(img['sizes'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {size}: {count} images")
    
    print("\nFile formats:")
    for fmt, count in img['formats'].items():
        print(f"  {fmt}: {count}")
    
    print("\nGrayscale statistics:")
    gray = img['gray_stats']
    print(f"  Min: {gray['min']}")
    print(f"  Max: {gray['max']}")
    print(f"  Average: {gray['avg']:.2f}")
    print(f"  Standard Deviation: {gray['std']:.2f}")
    
    print("\nCommon aspect ratios:")
    for ratio, count in sorted(img['aspect_ratios'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {ratio:.2f}: {count} images")

def main():
    datasets = [
        'data/raw/english_handwritten',
        'data/raw/emnist',
        'data/raw/mnist',
        'data/raw/az_handwritten',
        'data/raw/chars74k'
    ]
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            results = analyze_dataset(dataset_path)
            print_results(results)
            print("\n" + "="*50)
        else:
            print(f"\nPath {dataset_path} not found!")

if __name__ == "__main__":
        main()
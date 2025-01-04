from abc import ABC, abstractmethod
import os
import json
import pathlib
import cv2
import numpy as np
from typing import Dict, Any
from skimage.measure import shannon_entropy
from skimage.feature import hog
from scipy.stats import skew, kurtosis

class ImageAnalyzerInterface(ABC):
    @abstractmethod
    def analyze_first_images(self) -> Dict[str, Dict[str, Any]]:
        pass

class DatasetPath:
    BASE_PATH = "/Users/roammer/Documents/Github/handwriting/data/processed/images"
    
    @staticmethod
    def get_image_path() -> str:
        return DatasetPath.BASE_PATH

class ImageFeatureExtractor:
    @staticmethod
    def get_first_image_path(folder_path: str) -> str:
        try:
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)):
                    return os.path.join(folder_path, file)
        except Exception:
            return ""

    @staticmethod
    def count_images(folder_path: str) -> int:
        """Count total number of image files in the folder"""
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
            return sum(1 for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f)) 
                      and os.path.splitext(f)[1].lower() in image_extensions)
        except Exception:
            return 0

    @staticmethod
    def calculate_noise_ratio(image: np.ndarray) -> float:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        noise = cv2.subtract(image, blur)
        return float(np.mean(noise))

    @staticmethod
    def calculate_stroke_width(image: np.ndarray) -> float:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        return float(np.mean(dist[dist > 0]))

    @staticmethod
    def calculate_slant_angle(image: np.ndarray) -> float:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        if lines is not None:
            angles = [line[0][1] for line in lines]
            return float(np.median(angles) * 180/np.pi)
        return 0.0

    @staticmethod
    def calculate_contour_features(image: np.ndarray) -> Dict[str, float]:
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"contour_area": 0.0, "contour_perimeter": 0.0}
            
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        return {
            "contour_area": float(area),
            "contour_perimeter": float(perimeter)
        }

    @staticmethod
    def analyze_image(image_path: str) -> Dict[str, Any]:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {}

            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=False)

            features = {
                "dimensions": f"{image.shape[1]}x{image.shape[0]}",
                "noise_level": round(ImageFeatureExtractor.calculate_noise_ratio(image), 2),
                "mean_intensity": round(float(np.mean(gray)), 2),
                "std_intensity": round(float(np.std(gray)), 2),
                "entropy": round(float(shannon_entropy(gray)), 2),
                "stroke_width": round(ImageFeatureExtractor.calculate_stroke_width(gray), 2),
                "slant_angle": round(ImageFeatureExtractor.calculate_slant_angle(gray), 2),
                "skewness": round(float(skew(gray.flatten())), 2),
                "kurtosis": round(float(kurtosis(gray.flatten())), 2),
                "gradient_strength": round(float(np.mean(hog_features)), 3),
            }
            
            contour_features = ImageFeatureExtractor.calculate_contour_features(gray)
            features.update({k: round(v, 2) for k, v in contour_features.items()})
            
            return features
        except Exception as e:
            return {"error": str(e)}

class ImageAnalyzer(ImageAnalyzerInterface):
    def analyze_first_images(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        images_path = DatasetPath.get_image_path()
        first_image = ImageFeatureExtractor.get_first_image_path(images_path)
        
        results["processed"] = {}
        total_images = ImageFeatureExtractor.count_images(images_path)
        results["processed"]["total_images"] = total_images
        
        if first_image:
            features = ImageFeatureExtractor.analyze_image(first_image)
            results["processed"].update(features)
        else:
            results["processed"].update({"error": "No images found"})
                
        return results

class ResultWriter:
    @staticmethod
    def write_json(data: Dict[str, Dict[str, Any]]) -> None:
        current_file_path = pathlib.Path(__file__).parent
        output_path = current_file_path / "analyse_result.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

def main():
    analyzer = ImageAnalyzer()
    results = analyzer.analyze_first_images()
    ResultWriter.write_json(results)

if __name__ == "__main__":
    main()
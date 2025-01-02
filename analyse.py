import os
import cv2
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime
from PIL import Image, ImageStat
import imghdr
import struct
import hashlib
import magic
import chardet
from scipy import ndimage
from skimage import morphology, measure, filters, feature

@dataclass
class ImageMetrics:
    """Comprehensive image quality metrics"""
    # Basic properties
    dimensions: Tuple[int, int]
    file_format: str
    file_size: int
    actual_format: str  # Real format detected from magic numbers
    color_depth: int
    dpi: Optional[Tuple[float, float]]
    
    # Pixel statistics
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    median_intensity: float
    intensity_histogram: List[int]
    
    # Quality metrics
    contrast: float
    sharpness: float
    noise_level: float
    snr: float
    dynamic_range: float
    
    # Character specific metrics
    stroke_width_mean: float
    stroke_width_std: float
    char_background_ratio: float
    border_noise: float
    skew_angle: float
    aspect_ratio: float
    
    # Preprocessing detection
    is_binary: bool
    is_inverted: bool
    has_padding: bool
    normalization_range: Tuple[float, float]
    detected_preprocessing: List[str]

@dataclass
class DocumentationMetrics:
    """Comprehensive documentation analysis"""
    # File properties
    encoding: str
    line_endings: str
    total_lines: int
    file_size: int
    md5_hash: str
    
    # Content analysis
    fields: List[str]
    field_types: Dict[str, str]
    sample_entries: Dict[str, List[str]]
    value_ranges: Dict[str, Tuple[Any, Any]]
    unique_values: Dict[str, List[str]]
    
    # Quality checks
    missing_fields: List[str]
    invalid_entries: List[Dict[str, str]]
    formatting_issues: List[str]
    consistency_score: float

@dataclass
class DatasetStructure:
    """Dataset directory structure analysis"""
    total_files: int
    directory_tree: Dict[str, Any]
    file_naming_pattern: str
    file_extensions: Dict[str, int]
    folder_organization: str
    potential_issues: List[str]

class ComprehensiveAnalyzer:
    """Comprehensive dataset analyzer checking all quality aspects"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.output_file = self.project_root / "dataset_analysis.json"
        
        # Define comprehensive standards
        self.standards = {
            'image': {
                'size': (28, 28),
                'format': '.png',
                'color_mode': 'L',
                'dpi': 300,
                'bit_depth': 8,
                'max_file_size': 50_000,  # bytes
                'intensity_range': (0, 255),
                'min_contrast': 0.3,
                'max_noise': 0.1,
                'allowed_preprocessing': [
                    'normalization',
                    'centering',
                    'noise_reduction',
                    'binarization'
                ]
            },
            'documentation': {
                'required_fields': [
                    'image_id',
                    'label',
                    'writer_id',
                    'quality_score',
                    'collection_date',
                    'preprocessing_applied'
                ],
                'field_types': {
                    'image_id': 'str',
                    'label': 'str',
                    'writer_id': 'int',
                    'quality_score': 'float',
                    'collection_date': 'date',
                    'preprocessing_applied': 'list'
                },
                'encoding': 'utf-8',
                'line_endings': 'unix'
            },
            'structure': {
                'required_folders': ['images', 'metadata'],
                'max_depth': 3,
                'naming_pattern': r'^[a-zA-Z0-9_-]+$'
            }
        }

    def analyze_image(self, image_path: Path) -> ImageMetrics:
        """Comprehensive analysis of a single image"""
        # Read image in multiple ways for different checks
        cv_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        pil_img = Image.open(image_path)
        
        # Basic file properties
        file_stats = image_path.stat()
        real_format = magic.from_file(str(image_path), mime=True)
        
        # Get DPI if available
        try:
            dpi = pil_img.info.get('dpi')
        except:
            dpi = None
            
        # Pixel statistics
        stats = ImageStat.Stat(pil_img)
        hist = cv2.calcHist([cv_img], [0], None, [256], [0, 256]).flatten()
        
        # Quality metrics
        laplacian = cv2.Laplacian(cv_img, cv2.CV_64F)
        noise = cv2.subtract(cv_img, cv2.GaussianBlur(cv_img, (5,5), 0))
        
        # Character analysis
        _, binary = cv2.threshold(cv_img, 0, 255, cv2.THRESH_OTSU)
        distance = ndimage.distance_transform_edt(binary)
        skeleton = skimage.morphology.skeletonize(binary > 0)
        
        # Detect preprocessing
        preprocessing = []
        if np.array_equal(cv_img, binary):
            preprocessing.append('binarization')
        if np.abs(np.mean(cv_img) - 128) < 10:
            preprocessing.append('centering')
        if np.max(cv_img) <= 1.0:
            preprocessing.append('normalization')
        
        return ImageMetrics(
            dimensions=cv_img.shape,
            file_format=image_path.suffix,
            file_size=file_stats.st_size,
            actual_format=real_format,
            color_depth=pil_img.bits,
            dpi=dpi,
            
            mean_intensity=float(np.mean(cv_img)),
            std_intensity=float(np.std(cv_img)),
            min_intensity=float(np.min(cv_img)),
            max_intensity=float(np.max(cv_img)),
            median_intensity=float(np.median(cv_img)),
            intensity_histogram=hist.tolist(),
            
            contrast=float(np.max(cv_img) - np.min(cv_img)) / 255.0,
            sharpness=float(np.var(laplacian)),
            noise_level=float(np.std(noise)),
            snr=float(np.mean(cv_img) / np.std(cv_img)) if np.std(cv_img) != 0 else float('inf'),
            dynamic_range=float(np.log10(np.max(cv_img) / (np.min(cv_img) + 1e-6))),
            
            stroke_width_mean=float(np.mean(distance[distance > 0])),
            stroke_width_std=float(np.std(distance[distance > 0])),
            char_background_ratio=float(np.sum(binary > 0) / binary.size),
            border_noise=float(np.mean(cv_img[0]) + np.mean(cv_img[-1]) + 
                             np.mean(cv_img[:,0]) + np.mean(cv_img[:,-1])) / 4,
            skew_angle=float(measure.regionprops(binary.astype(int))[0].orientation),
            aspect_ratio=float(binary.shape[1] / binary.shape[0]),
            
            is_binary=len(np.unique(cv_img)) <= 2,
            is_inverted=np.mean(cv_img) > 127,
            has_padding=float(np.mean(cv_img[:5]) + np.mean(cv_img[-5:])) < 10,
            normalization_range=(float(np.min(cv_img)), float(np.max(cv_img))),
            detected_preprocessing=preprocessing
        )

    def analyze_documentation(self, doc_path: Path) -> DocumentationMetrics:
        """Comprehensive documentation file analysis"""
        with open(doc_path, 'rb') as f:
            content = f.read()
            
        # File properties
        encoding = chardet.detect(content)['encoding']
        text_content = content.decode(encoding)
        lines = text_content.splitlines()
        
        # Detect line endings
        line_ending = 'unix'
        if '\r\n' in text_content:
            line_ending = 'windows'
        elif '\r' in text_content:
            line_ending = 'mac'
            
        # Content analysis
        header = lines[0].strip().split('\t')
        data_sample = [line.strip().split('\t') for line in lines[1:6]]  # First 5 entries
        
        # Field analysis
        field_types = {}
        value_ranges = {}
        unique_values = {field: set() for field in header}
        invalid_entries = []
        
        for row in data_sample:
            for field, value in zip(header, row):
                unique_values[field].add(value)
                
                # Detect field type and validate
                try:
                    int(value)
                    field_types[field] = 'int'
                except:
                    try:
                        float(value)
                        field_types[field] = 'float'
                    except:
                        field_types[field] = 'str'
                        
                # Check for invalid entries
                if not value.strip():
                    invalid_entries.append({
                        'field': field,
                        'value': value,
                        'reason': 'empty'
                    })
                    
        # Calculate consistency score
        expected_fields = set(self.standards['documentation']['required_fields'])
        found_fields = set(header)
        consistency_score = len(expected_fields & found_fields) / len(expected_fields)
        
        return DocumentationMetrics(
            encoding=encoding,
            line_endings=line_ending,
            total_lines=len(lines),
            file_size=len(content),
            md5_hash=hashlib.md5(content).hexdigest(),
            
            fields=header,
            field_types=field_types,
            sample_entries={field: list(values)[:5] for field, values in unique_values.items()},
            value_ranges={},  # Filled based on field types
            unique_values={field: list(values) for field, values in unique_values.items()},
            
            missing_fields=list(expected_fields - found_fields),
            invalid_entries=invalid_entries,
            formatting_issues=[],  # Filled based on specific checks
            consistency_score=consistency_score
        )

    def analyze_dataset_structure(self, dataset_path: Path) -> DatasetStructure:
        """Analyze dataset directory structure"""
        total_files = 0
        file_extensions = {}
        directory_tree = {}
        issues = []
        
        for root, dirs, files in os.walk(dataset_path):
            rel_path = str(Path(root).relative_to(dataset_path))
            directory_tree[rel_path] = {
                'dirs': dirs,
                'files': files,
                'depth': len(Path(rel_path).parts)
            }
            
            total_files += len(files)
            
            # Check file extensions
            for file in files:
                ext = Path(file).suffix
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
                
                # Check naming pattern
                if not file.replace('.', '').replace('-', '').isalnum():
                    issues.append(f"Invalid filename: {file}")
            
            # Check directory depth
            if len(Path(root).parts) > self.standards['structure']['max_depth']:
                issues.append(f"Directory too deep: {root}")
        
        # Determine folder organization pattern
        if 'images' in directory_tree.get('', {}).get('dirs', []):
            organization = 'flat'
        else:
            organization = 'hierarchical'
            
        return DatasetStructure(
            total_files=total_files,
            directory_tree=directory_tree,
            file_naming_pattern=self._detect_naming_pattern(dataset_path),
            file_extensions=file_extensions,
            folder_organization=organization,
            potential_issues=issues
        )

    def _detect_naming_pattern(self, dataset_path: Path) -> str:
        """Detect file naming pattern in dataset"""
        sample_files = list(dataset_path.rglob('*'))[:5]
        if not sample_files:
            return 'unknown'
            
        patterns = []
        for file in sample_files:
            name = file.stem
            pattern = ''
            for char in name:
                if char.isdigit():
                    pattern += '#'
                elif char.isalpha():
                    pattern += 'a'
                else:
                    pattern += char
            patterns.append(pattern)
            
        # Return most common pattern
        return max(set(patterns), key=patterns.count)

    def analyze_dataset(self, dataset_name: str) -> Dict:
        """Analyze a single dataset comprehensively"""
        dataset_path = self.project_root / "data" / "temp" / dataset_name
        
        # Get first image for analysis
        images_path = dataset_path / "images"
        sample_image = next(images_path.glob('*'))
        
        analysis = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            
            # Structural analysis
            'structure': asdict(self.analyze_dataset_structure(dataset_path)),
            
            # Documentation analysis
            'documentation': asdict(self.analyze_documentation(dataset_path / "documentation.txt")),
            
            # Sample image analysis
            'sample_image': {
                'path': str(sample_image),
                'metrics': asdict(self.analyze_image(sample_image)),
                'meets_standards': self._check_image_standards(sample_image)
            },
            
            # Dataset-wide statistics
            'statistics': {
                'total_images': len(list(images_path.glob('*'))),
                'total_size': sum(f.stat().st_size for f in images_path.glob('*')),
                'class_distribution': self._get_class_distribution(images_path)
            }
        }
        
        # Add compliance score
        analysis['compliance_score'] = self._calculate_compliance_score(analysis)
        
        # Add specific recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis

    def _check_image_standards(self, image_path: Path) -> Dict[str, bool]:
        """Check if image meets all defined standards"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return {
            'size_ok': img.shape == self.standards['image']['size'],
            'format_ok': image_path.suffix == self.standards['image']['format'],
            'size_under_limit': image_path.stat().st_size <= self.standards['image']['max_file_size'],
            'bit_depth_ok': Image.open(image_path).bits == self.standards['image']['bit_depth'],
            'contrast_ok': (np.max(img) - np.min(img)) / 255.0 >= self.standards['image']['min_contrast'],
            'noise_ok': np.std(cv2.subtract(img, cv2.GaussianBlur(img, (5,5), 0))) <= self.standards['image']['max_noise']
        }

    def _get_class_distribution(self, images_path: Path) -> Dict[str, int]:
        """Get distribution of classes based on filename prefixes"""
        distribution = {}
        for image in images_path.glob('*'):
            class_name = image.stem.split('_')[0]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def _calculate_compliance_score(self, analysis: Dict) -> float:
        """Calculate overall compliance score for dataset"""
        scores = []
        
        # Image standards compliance
        img_standards = analysis['sample_image']['meets_standards']
        scores.append(sum(img_standards.values()) / len(img_standards))
        
        # Documentation compliance
        doc_score = analysis['documentation']['consistency_score']
        scores.append(doc_score)
        
        # Structure compliance
        struct_score = 1.0
        if analysis['structure']['potential_issues']:
            struct_score -= len(analysis['structure']['potential_issues']) * 0.1
        scores.append(max(0, struct_score))
        
        return sum(scores) / len(scores)

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate specific recommendations for dataset improvement"""
        recommendations = []
        
        # Image-related recommendations
        img_standards = analysis['sample_image']['meets_standards']
        metrics = analysis['sample_image']['metrics']
        
        if not img_standards['size_ok']:
            recommendations.append(f"Resize images to {self.standards['image']['size']}")
        if not img_standards['format_ok']:
            recommendations.append(f"Convert images to {self.standards['image']['format']}")
        if not img_standards['contrast_ok']:
            recommendations.append("Enhance image contrast")
        if not img_standards['noise_ok']:
            recommendations.append("Apply noise reduction")
            
        # Documentation-related recommendations
        if analysis['documentation']['missing_fields']:
            recommendations.append(f"Add missing documentation fields: {', '.join(analysis['documentation']['missing_fields'])}")
        if analysis['documentation']['invalid_entries']:
            recommendations.append("Clean invalid documentation entries")
            
        # Structure-related recommendations
        if analysis['structure']['potential_issues']:
            recommendations.append("Fix directory structure issues")
            
        return recommendations

    def run_complete_analysis(self):
        """Run comprehensive analysis on all datasets"""
        start_time = datetime.now()
        
        # Analyze each dataset
        datasets = ['EH', 'EMNIST', 'MNIST', 'AZ', 'Chars74k']
        results = {
            'analysis_date': start_time.isoformat(),
            'datasets': {},
            'compatibility_matrix': {},
            'global_recommendations': []
        }
        
        # Individual dataset analysis
        for dataset in datasets:
            try:
                results['datasets'][dataset] = self.analyze_dataset(dataset)
            except Exception as e:
                results['datasets'][dataset] = {'error': str(e)}
                
        # Generate compatibility matrix
        for i, dataset1 in enumerate(datasets):
            if dataset1 not in results['compatibility_matrix']:
                results['compatibility_matrix'][dataset1] = {}
            for dataset2 in datasets[i+1:]:
                compatibility = self._check_datasets_compatibility(
                    results['datasets'][dataset1],
                    results['datasets'][dataset2]
                )
                results['compatibility_matrix'][dataset1][dataset2] = compatibility
                
        # Generate global recommendations
        results['global_recommendations'] = self._generate_global_recommendations(results)
        
        # Add execution summary
        end_time = datetime.now()
        results['execution_summary'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'datasets_analyzed': len(datasets),
            'successful_analyses': sum(1 for d in results['datasets'].values() if 'error' not in d)
        }
        
        # Save results to JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        return results

    def _check_datasets_compatibility(self, dataset1: Dict, dataset2: Dict) -> Dict:
        """Check compatibility between two datasets"""
        if 'error' in dataset1 or 'error' in dataset2:
            return {'error': 'One or both datasets have analysis errors'}
            
        compatibility = {
            'image_format_compatible': dataset1['sample_image']['metrics']['file_format'] == 
                                     dataset2['sample_image']['metrics']['file_format'],
            'image_size_compatible': dataset1['sample_image']['metrics']['dimensions'] ==
                                   dataset2['sample_image']['metrics']['dimensions'],
            'preprocessing_compatible': set(dataset1['sample_image']['metrics']['detected_preprocessing']) ==
                                     set(dataset2['sample_image']['metrics']['detected_preprocessing']),
            'quality_difference': abs(dataset1['compliance_score'] - dataset2['compliance_score']),
            'documentation_compatible': self._check_documentation_compatibility(
                dataset1['documentation'],
                dataset2['documentation']
            ),
            'class_overlap': self._calculate_class_overlap(
                dataset1['statistics']['class_distribution'],
                dataset2['statistics']['class_distribution']
            ),
            'merge_complexity': 'low'  # Will be updated based on checks
        }
        
        # Determine merge complexity
        issues = []
        if not compatibility['image_format_compatible']:
            issues.append("format conversion needed")
        if not compatibility['image_size_compatible']:
            issues.append("resizing needed")
        if not compatibility['preprocessing_compatible']:
            issues.append("preprocessing standardization needed")
        if compatibility['quality_difference'] > 0.2:
            issues.append("quality normalization needed")
            
        if len(issues) >= 3:
            compatibility['merge_complexity'] = 'high'
        elif len(issues) >= 1:
            compatibility['merge_complexity'] = 'medium'
            
        compatibility['merge_issues'] = issues
        
        return compatibility

    def _check_documentation_compatibility(self, doc1: Dict, doc2: Dict) -> bool:
        """Check if two documentation formats are compatible"""
        return (
            set(doc1['fields']) == set(doc2['fields']) and
            doc1['field_types'] == doc2['field_types'] and
            doc1['encoding'] == doc2['encoding']
        )

    def _calculate_class_overlap(self, dist1: Dict[str, int], dist2: Dict[str, int]) -> float:
        """Calculate overlap between class distributions"""
        classes1 = set(dist1.keys())
        classes2 = set(dist2.keys())
        return len(classes1 & classes2) / len(classes1 | classes2)

    def _generate_global_recommendations(self, results: Dict) -> List[str]:
        """Generate global recommendations for all datasets"""
        recommendations = []
        
        # Check format standardization
        formats = set()
        for dataset in results['datasets'].values():
            if 'error' not in dataset:
                formats.add(dataset['sample_image']['metrics']['file_format'])
        if len(formats) > 1:
            recommendations.append(f"Standardize image format across all datasets to {self.standards['image']['format']}")
            
        # Check size standardization
        sizes = set()
        for dataset in results['datasets'].values():
            if 'error' not in dataset:
                sizes.add(tuple(dataset['sample_image']['metrics']['dimensions']))
        if len(sizes) > 1:
            recommendations.append(f"Standardize image size across all datasets to {self.standards['image']['size']}")
            
        # Check preprocessing consistency
        preprocessing_sets = []
        for dataset in results['datasets'].values():
            if 'error' not in dataset:
                preprocessing_sets.append(
                    set(dataset['sample_image']['metrics']['detected_preprocessing'])
                )
        if len(set(map(frozenset, preprocessing_sets))) > 1:
            recommendations.append("Standardize preprocessing steps across all datasets")
            
        # Check documentation consistency
        field_sets = []
        for dataset in results['datasets'].values():
            if 'error' not in dataset:
                field_sets.append(set(dataset['documentation']['fields']))
        if len(set(map(frozenset, field_sets))) > 1:
            recommendations.append("Standardize documentation fields across all datasets")
            
        # Add quality-based recommendations
        low_quality_datasets = [
            name for name, dataset in results['datasets'].items()
            if 'error' not in dataset and dataset['compliance_score'] < 0.8
        ]
        if low_quality_datasets:
            recommendations.append(
                f"Improve quality for datasets: {', '.join(low_quality_datasets)}"
            )
            
        return recommendations


def main():
    # Set project root
    project_root = Path(__file__).parent.resolve()
    
    # Initialize and run analyzer
    analyzer = ComprehensiveAnalyzer(project_root)
    results = analyzer.run_complete_analysis()
    
    print(f"Analysis completed. Results saved to: {analyzer.output_file}")


if __name__ == "__main__":
    main()
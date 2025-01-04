import logging
from typing import Dict, Type
from pathlib import Path
from src.lib.cache_manager import CacheManager
from src.services.enhancers.base_enhancer import BaseEnhancer
from src.services.enhancers.brightness_enhancer import BrightnessEnhancer
from src.services.enhancers.noise_enhancer import NoiseEnhancer
from src.services.enhancers.stroke_enhancer import StrokeEnhancer
from src.services.enhancers.quality_enhancer import QualityEnhancer
from src.services.enhancers.data_splitter import DataSplitter
# from src.services.enhancers.merger import Merger

def setup_basic_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class EnhancementPipeline:
    DATASET_NAMES = ["EH", "MNIST", "AZ", "Chars74K"]
    TEMP_PATH = Path("/Users/roammer/Documents/Github/handwriting/data/temp")
    
    def __init__(self):
        self.enhancers: Dict[str, Type[BaseEnhancer]] = {
            "noise": NoiseEnhancer,        
            # "stroke": StrokeEnhancer,        
            # "brightness": BrightnessEnhancer, 
            # "quality": QualityEnhancer,      
            # "merger": Merger,
            # "splitter": DataSplitter,
        }
        
    def process_dataset(self, dataset_name: str) -> bool:
        try:
            dataset_path = self.TEMP_PATH / dataset_name / "images"
            logging.info(f"Starting enhancement pipeline for {dataset_name}")
            
            # Apply each enhancement step
            for enhancer_name, enhancer_class in self.enhancers.items():
                logging.info(f"Applying {enhancer_name} enhancement to {dataset_name}")
                enhancer = enhancer_class(dataset_path)
                if not enhancer.process():
                    logging.error(f"Failed to apply {enhancer_name} to {dataset_name}")
                    return False
                
            logging.info(f"Successfully completed enhancement pipeline for {dataset_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            return False

def main() -> bool:
    try:
        # Setup logging
        setup_basic_logging()
        logging.info("Starting data enhancement pipeline")
        
        # Initialize pipeline
        pipeline = EnhancementPipeline()
        
        # Process each dataset
        success = True
        for dataset_name in EnhancementPipeline.DATASET_NAMES:
            if not pipeline.process_dataset(dataset_name):
                logging.error(f"Failed to enhance {dataset_name} dataset")
                success = False
        
        return success
            
    except Exception as e:
        logging.error(f"Critical error in enhancement pipeline: {str(e)}")
        return False
    finally:
        # Clean cache and temporary files
        CacheManager.cleanup()
        logging.info("Pipeline completed. Cache cleaned up.")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
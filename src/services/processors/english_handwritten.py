import logging
import shutil
from pathlib import Path
from typing import Optional

class EnglishHandwrittenProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "english_handwritten"
        self.temp_path = self.project_root / "data" / "temp" / "EH"
        
    def process(self) -> bool:
        try:
            self.logger.info("Processing English handwritten text...")
            
            # Create temp directory if it doesn't exist
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            # Copy dataset to temp location
            if self.source_path.exists():
                self._copy_dataset()
                self.logger.info(f"Dataset copied to {self.temp_path}")
                return True
            else:
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in English handwritten processing: {str(e)}")
            return False
            
    def _copy_dataset(self) -> None:
        """Copy dataset files to temp directory"""
        if self.source_path.exists():
            # Copy entire directory tree
            shutil.copytree(
                self.source_path, 
                self.temp_path, 
                dirs_exist_ok=True
            )
            self.logger.info(f"Copied dataset from {self.source_path} to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")
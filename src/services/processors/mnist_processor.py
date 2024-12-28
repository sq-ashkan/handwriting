import logging
import shutil
from pathlib import Path

class MNISTProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.source_path = self.project_root / "data" / "raw" / "mnist"
        self.temp_path = self.project_root / "data" / "temp" / "MNIST"
        
    def process(self) -> bool:
        try:
            self.logger.info("Processing MNIST dataset...")
            self.temp_path.mkdir(parents=True, exist_ok=True)
            
            if self.source_path.exists():
                self._copy_dataset()
                self.logger.info(f"Dataset copied to {self.temp_path}")
                return True
            else:
                self.logger.error(f"Source dataset not found at {self.source_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error in MNIST processing: {str(e)}")
            return False
            
    def _copy_dataset(self) -> None:
        if self.source_path.exists():
            shutil.copytree(
                self.source_path, 
                self.temp_path, 
                dirs_exist_ok=True
            )
            self.logger.info(f"Copied dataset from {self.source_path} to {self.temp_path}")
        else:
            raise FileNotFoundError(f"Source dataset not found at {self.source_path}")
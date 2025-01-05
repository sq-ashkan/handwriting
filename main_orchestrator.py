import os
import time
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

class MainOrchestrator:
    @staticmethod
    def execute_pipeline():
        commands = [
            "PYTHONPATH=$PYTHONPATH:. python src/operators/main_downloader.operator.py",
            "PYTHONPATH=$PYTHONPATH:. python src/operators/main_processor.operator.py",
            "PYTHONPATH=$PYTHONPATH:. python src/operators/main_data_enhancers.operator.py", 
            "PYTHONPATH=$PYTHONPATH:. python src/operators/main_modifier.operator.py",
            "PYTHONPATH=$PYTHONPATH:. python src/operators/main_augmentation.operator.py",
            "PYTHONPATH=$PYTHONPATH:. python src/model/train.operator.py",
            "PYTHONPATH=$PYTHONPATH:. python src/api/api.operator.py"
        ]

        for cmd in commands:
            try:
                if os.system(cmd) != 0:
                    logging.error(f"Failed: {cmd}")
                    return False
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error: {e}")
                return False
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = MainOrchestrator.execute_pipeline()
    exit(0 if success else 1)
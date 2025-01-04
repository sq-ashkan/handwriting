import os
import time
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

class MainOrchestrator:
    @staticmethod
    def create_temp_copies():
        data_path = Path('data')
        temp_path = data_path / 'temp'
        if temp_path.exists():
            for i in tqdm(range(1, 11), desc="Creating temp copies", ncols=100):
                new_temp = data_path / f'temp{i}'
                if not new_temp.exists():
                    shutil.copytree(temp_path, new_temp)
                time.sleep(0.1)  # Small delay for visible progress bar

    @staticmethod
    def execute_pipeline():
        commands = [
            # "python main_downloader.py",
            # "python main_processor.py",
            "python analyse.py",
            "mv analyse_result.json analyse_result_before.json",
            "python main_data_enhancers.py",
            "python analyse.py"
        ]

        for cmd in commands:
            try:
                if os.system(cmd) != 0:
                    logging.error(f"Failed: {cmd}")
                    return False
                if cmd == "python main_processor.py":
                    MainOrchestrator.create_temp_copies()
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error: {e}")
                return False
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = MainOrchestrator.execute_pipeline()
    exit(0 if success else 1)
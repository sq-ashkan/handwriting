import os
import time
from pathlib import Path
import logging

class MainOrchestrator:
    @staticmethod
    def execute_pipeline():
        commands = [
            "python main_processor.py",
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
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error: {e}")
                return False
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = MainOrchestrator.execute_pipeline()
    exit(0 if success else 1)
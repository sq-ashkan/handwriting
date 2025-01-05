import sys
import logging
from pathlib import Path
from .constants import LOGS_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")
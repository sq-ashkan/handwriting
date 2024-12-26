import sys
import shutil
import logging
from pathlib import Path
from src.lib.utils import setup_logging
from src.services.iam_downloader import IAMDatasetDownloader
from src.services.emnist_downloader import EMNISTDatasetDownloader
from src.services.mnist_downloader import MNISTDatasetDownloader
from src.lib.constants import RAW_DIR

def cleanup_pycache():
    project_root = Path(__file__).parent
    for pycache_dir in project_root.rglob("__pycache__"):
        shutil.rmtree(pycache_dir)

def main() -> bool:
    try:
        setup_logging()
        
        logging.info("Starting IAM download")
        iam_downloader = IAMDatasetDownloader()
        iam_success = iam_downloader.run()
        
        logging.info("Starting EMNIST download")
        emnist_downloader = EMNISTDatasetDownloader()
        emnist_success = emnist_downloader.run()
        
        logging.info("Starting MNIST download")
        mnist_downloader = MNISTDatasetDownloader()
        mnist_success = mnist_downloader.run()
        
        cleanup_pycache()
        
        all_success = iam_success and emnist_success and mnist_success
        
        if all_success:
            logging.info("All datasets downloaded successfully")
        else:
            logging.error("Failed to download some datasets")
            
        return all_success
        
    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nDataset files location: {RAW_DIR}")
    sys.exit(0 if success else 1)
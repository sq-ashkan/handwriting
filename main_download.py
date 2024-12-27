import sys
import logging
from pathlib import Path
from src.lib.utils import setup_logging
from src.lib.cache_manager import CacheManager
from src.services.english_handwritten import EnglishHandwrittenDownloader
from src.services.emnist_downloader import EMNISTDatasetDownloader
from src.services.mnist_downloader import MNISTDatasetDownloader
from src.services.az_downloader import AZDatasetDownloader
from src.services.chars74k_downloader import Chars74KDatasetDownloader
from src.lib.constants import RAW_DIR

def main() -> bool:
    try:
        # پاک‌سازی کش قبل از شروع
        CacheManager.cleanup()
        
        # راه‌اندازی سیستم لاگ
        setup_logging()
        
        # تعریف دانلودرها
        downloaders = {
            "EH": EnglishHandwrittenDownloader(),
            "EMNIST": EMNISTDatasetDownloader(),
            "MNIST": MNISTDatasetDownloader(),
            "A-Z": AZDatasetDownloader(),
            "Chars74K": Chars74KDatasetDownloader()
        }
        
        success_status = {}
        
        for name, downloader in downloaders.items():
            logging.info(f"Starting {name} dataset download...")
            try:
                success_status[name] = downloader.run()
                if success_status[name]:
                    logging.info(f"{name} dataset successfully downloaded and processed")
                else:
                    logging.error(f"{name} dataset processing failed")
            except Exception as e:
                logging.error(f"Error with {name} dataset: {str(e)}")
                success_status[name] = False
        
        all_success = all(success_status.values())
        
        if all_success:
            logging.info("All datasets successfully downloaded and processed")
        else:
            failed = [name for name, success in success_status.items() if not success]
            logging.error(f"Failed datasets: {', '.join(failed)}")
        
        logging.info(f"\nDataset location: {RAW_DIR}")
        for name, success in success_status.items():
            status = "✓" if success else "✗"
            logging.info(f"{status} {name}")
        
        return all_success
        
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
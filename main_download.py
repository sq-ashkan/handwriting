import sys
from src.lib.utils import setup_logging
from src.services.iam_downloader import IAMDatasetDownloader
from src.services.emnist_downloader import EMNISTDatasetDownloader  # اضافه شد
from src.lib.constants import RAW_DIR

def main() -> bool:
    try:
        setup_logging()
        
        # دانلود IAM
        # iam_downloader = IAMDatasetDownloader()
        # iam_success = iam_downloader.run()
        
        # دانلود EMNIST
        emnist_downloader = EMNISTDatasetDownloader()
        emnist_success = emnist_downloader.run()
        
        return emnist_success
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    print("\nDataset files location:", RAW_DIR)
    sys.exit(0 if success else 1)
import logging
from src.services.mergers.merger import merge
from src.services.mergers.verifier import verify
from src.services.mergers.splitter import split
from src.services.mergers.cleaner import cleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # merge()
        # verify()
        # split()
        cleaner()
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
    
    exit(0)
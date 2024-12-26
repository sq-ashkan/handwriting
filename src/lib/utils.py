import sys
import logging
from pathlib import Path
from .constants import LOGS_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR

def setup_logging():
    # Function to initialize the logging system
    
    # Define path and filename for logging
    log_file = LOGS_DIR / "download.log"
    
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Basic logging configuration
    logging.basicConfig(
        # Set logging level - captures all important messages
        level=logging.INFO,
        
        # Format for log messages:
        # timestamp - level - message
        format='%(asctime)s - %(levelname)s - %(message)s',
        
        # Define log output destinations:
        handlers=[
            # Save to specified file
            logging.FileHandler(log_file),
            
            # Display in program output
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    # Function to create necessary project directories
    
    # Loop through all paths defined in settings
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
        
        # Create each directory with two properties:
        # parents=True: creates parent directories if they don't exist
        # exist_ok=True: doesn't raise error if directory exists
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log the creation of each directory
        logging.info(f"Created directory: {dir_path}")
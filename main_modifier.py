import logging
import os
import time
from src.services.mergers.merger import merge
from src.services.mergers.verifier import verify
from src.services.mergers.splitter import split
from src.services.mergers.cleaner import cleaner
from src.services.mergers.finalizer import finalizer
from src.services.mergers.cache_cleaner import cache_cleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_terminal():
    """Clear terminal screen based on operating system"""
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For Unix/Linux/MacOS
    else:
        os.system('clear')

def print_final_message():
    """Display the final success message"""
    clear_terminal()
    print("\n" + "=" * 50)
    print("\n✨ Ash_500k_proccessed_data created successfully!")
    print("\nAll processing steps completed.")
    print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    try:
        # merge()
        # verify()
        # split()
        # cleaner()
        finalizer()
        cache_cleaner()
        
        # Wait briefly to ensure all logs are displayed
        time.sleep(1)
        print_final_message()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
    
    exit(0)
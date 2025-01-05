from pathlib import Path
import shutil
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def finalizer():
    try:

        base = Path('/Users/roammer/Documents/Github/handwriting/data')
        processed = base / 'processed'
        target = base / 'Ash_500k_proccessed_data'

        if not processed.exists():
            logger.error("Processed directory not found. Cleaner must run first.")
            return False

        logger.info("Starting final processing steps...")


        target.mkdir(exist_ok=True)


        main_folders = ['digits', 'lowercase', 'uppercase']
        

        for folder in tqdm(main_folders, desc="Processing main folders"):
            source_folder = processed / folder
            
            if not source_folder.exists():
                logger.warning(f"Source folder not found: {folder}")
                continue

  
            subfolders = list(source_folder.glob('*'))
            for subfolder in tqdm(subfolders, desc=f"Processing {folder} subfolders"):
                if subfolder.is_dir():
                    new_folder = target / subfolder.name
                    new_folder.mkdir(exist_ok=True)
                    
                   
                    for item in subfolder.glob('*'):
                        if item.is_file():
                            shutil.copy2(item, new_folder)
                        elif item.is_dir():
                            shutil.copytree(item, new_folder / item.name, dirs_exist_ok=True)
                    
                    logger.info(f"Processed: {subfolder.name}")

        logger.info("Finalization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during finalization: {e}")
        return False

if __name__ == "__main__":
    finalizer()
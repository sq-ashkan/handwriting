from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

def ensure_character_folder(ready_path: Path, category: str, label: str) -> Path:
    """Create folder for a specific character category if it doesn't exist"""
    char_path = ready_path / category / label
    
    if not char_path.exists():
        # Create folder structure
        char_path.mkdir(parents=True)
        (char_path / 'images').mkdir()
        # Create empty documentation file
        (char_path / 'documentation.txt').touch()
        logger.info(f"Created new folder structure for {category}/{label}")
        
    return char_path

def get_character_category(label: str) -> tuple[str, bool]:
    """Determine the category of a character and its validity"""
    if len(label) != 1:
        return "", False
        
    if label.isalpha():
        if label.isupper():
            return "uppercase", True
        else:
            return "lowercase", True
    elif label.isdigit():
        return "digits", True
        
    return "", False

def split():
    """Main function to split dataset by character and case sensitivity"""
    logger.info("Starting dataset split process")
    
    try:
        # Setup paths
        project_base = Path('/Users/roammer/Documents/Github/handwriting')
        ready_path = project_base / 'data' / 'ready'
        final_path = project_base / 'data' / 'final'
        
        # Create base ready folder
        ready_path.mkdir(parents=True, exist_ok=True)
        
        # Create category folders
        for category in ['uppercase', 'lowercase', 'digits']:
            (ready_path / category).mkdir(exist_ok=True)
        
        # Read and process documentation file
        doc_path = final_path / 'documentation.txt'
        
        # Keep track of statistics
        stats = {
            'uppercase': 0,
            'lowercase': 0,
            'digits': 0,
            'errors': 0
        }
        
        with open(doc_path, 'r') as doc_file:
            for line_num, line in enumerate(doc_file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Split line into image name and label
                    parts = line.split()
                    if len(parts) != 2:
                        logger.error(f"Invalid line format at line {line_num}: {line}")
                        stats['errors'] += 1
                        continue
                        
                    image_name, label = parts
                    
                    # Determine character category
                    category, is_valid = get_character_category(label)
                    
                    if not is_valid:
                        logger.error(f"Invalid label at line {line_num}: {label}")
                        stats['errors'] += 1
                        continue
                    
                    # Create or get character folder
                    char_folder = ensure_character_folder(ready_path, category, label)
                    
                    # Copy image
                    source_image = final_path / 'images' / f"{image_name}.png"
                    dest_image = char_folder / 'images' / f"{image_name}.png"
                    
                    if not source_image.exists():
                        logger.error(f"Source image not found: {source_image}")
                        stats['errors'] += 1
                        continue
                    
                    shutil.copy2(source_image, dest_image)
                    
                    # Append to character documentation
                    with open(char_folder / 'documentation.txt', 'a') as char_doc:
                        char_doc.write(f"{image_name} {label}\n")
                    
                    # Update statistics
                    stats[category] += 1
                    
                    logger.info(f"Processed image {image_name} for {category}/{label}")
                    
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {str(e)}")
                    stats['errors'] += 1
                    continue
        
        # Log final statistics
        logger.info("Dataset split completed successfully")
        logger.info(f"Statistics: Uppercase: {stats['uppercase']}, "
                   f"Lowercase: {stats['lowercase']}, "
                   f"Digits: {stats['digits']}, "
                   f"Errors: {stats['errors']}")
        
    except Exception as e:
        logger.error(f"Error in split process: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    split()
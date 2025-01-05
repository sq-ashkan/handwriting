from pathlib import Path
from shutil import rmtree

def cleaner():
    base = Path('/Users/roammer/Documents/Github/handwriting/data')
    
    folders_to_clean = ['temp', 'raw', 'processed', 'final']
    for folder_name in folders_to_clean:
        folder = base / folder_name
        if folder.exists():
            rmtree(folder)
    
    ready = base / 'ready'
    if ready.exists():
        ready.rename(base / 'processed')
        print("✅ Folder 'ready' renamed to 'processed'.")
    else:
        print("Folder 'ready' not found.")

if __name__ == "__main__":
    cleaner()
from pathlib import Path
import shutil
from tqdm import tqdm

def cleaner():
    base = Path('/Users/roammer/Documents/Github/handwriting/data')
    ready = base / 'ready'

    # 1. پاک کردن فولدرهای temp، raw، processed و final
    for dir in tqdm(['temp', 'raw', 'processed', 'final'], desc="1. Cleaning Folders"):
        folder = base / dir
        if folder.exists():
            shutil.rmtree(folder)

    # 2. تغییر نام پوشه ready به processed
    if ready.exists():
        ready.rename(base / 'processed')
        print("\n✅ Folder 'ready' renamed to 'processed'.")
    else:
        print("\n❌ Folder 'ready' not found.")

if __name__ == "__main__":
    cleaner()
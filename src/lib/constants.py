from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Adding new directory paths
IAM_DIR = RAW_DIR / "iam_handwriting"
EMNIST_DIR = RAW_DIR / "emnist"
MNIST_DIR = RAW_DIR / "mnist"

PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
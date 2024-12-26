# Handwriting Recognition System Documentation

## Environment Setup
This project uses Conda for environment management. Here's how to set up:

```bash
# Create new conda environment
conda create -n handwriting python=3.10

# Activate environment
conda activate handwriting

# Install requirements
pip install -r requirements.txt
```

## Project Structure

### Main Entry Points
The project is divided into four main entry points:

1. `main_download.py`: Downloads IAM Handwriting dataset from Kaggle
2. `main_preprocess.py`: Handles image preprocessing and augmentation
3. `main_train.py`: Manages model training and validation
4. `main_app_launcher.py`: Launches Flask API for inference

### Requirements
```text
kagglehub>=0.1.12
opencv-python>=4.8.0
torch>=2.1.0
torchvision>=0.16.0
pytorch-lightning>=2.1.0
pathlib>=1.0.1
logging>=0.5.1.2
```

## Detailed Download Process

### File Structure
```
project_root/
├── src/
│   ├── main_download.py
│   ├── main_preprocess.py
│   ├── main_train.py
│   ├── main_app_launcher.py
│   ├── services/
│   │   ├── downloader.py
│   │   ├── preprocessor.py
│   │   └── trainer.py
│   └── lib/
│       ├── utils.py
│       └── constants.py
├── data/
│   ├── raw/             # Original downloaded data
│   ├── processed/       # Preprocessed data
│   └── logs/           # Log files
├── requirements.txt
└── README.md
```

### Download Process Details
The download process is handled by `main_download.py` and consists of the following steps:

1. **Directory Creation**
   - Creates necessary directories (raw, processed, logs)
   - Sets up logging system

2. **Dataset Download**
   - Downloads from Kaggle using kagglehub
   - Source: "naderabdalghani/iam-handwritten-forms-dataset"
   - Temporary storage in system temp directory

3. **File Movement**
   - Moves downloaded files to raw directory
   - Maintains original file structure
   - Preserves file metadata

4. **Verification**
   - Checks for required directories and files
   - Validates download integrity
   - Logs success or failure

### Storage Requirements
- Raw dataset size: ~900MB
- Required free space: 2GB minimum
- Processed data may require additional 1-2GB

### Running Download Process
```bash
# Activate conda environment
conda activate handwriting

# Run download script
python src/main_download.py
```

### Download Output
The download process creates the following structure in the data directory:
```
data/
├── raw/
│   └── data/           
│       ├── 0/          # Images of digit 0
│       ├── 1/          # Images of digit 1
│       ├── 2/          # Images of digit 2
│       ...
│       └── 9/          # Images of digit 9
├── logs/
│   └── download.log    # Download process logs
```

### Dataset Details
- The downloaded dataset contains separate folders for each digit (0-9)
- Each folder contains multiple handwritten samples of that digit
- Images are in PNG format
- Total size: ~900MB
- Number of samples: 671

### Important Notes
1. Dataset Structure:
   - Organized by class (digits 0-9)
   - Each class has its own directory
   - Makes it easier for training classification models

2. For Preprocessing:
   - We'll need to handle each digit folder separately
   - Create train/validation splits maintaining class distribution
   - Implement balanced sampling during training

3. Storage Considerations:
   - Original data in raw/data/[0-9]/
   - Processed data will maintain similar structure
   - Each preprocessing step will be tracked in logs

### Error Handling
- Network issues are logged and retried
- Incomplete downloads are detected and restarted
- Verification failures trigger clear error messages
- All errors are logged with full stack traces

## Next Steps
After successful download:
1. Run preprocessing (`main_preprocess.py`)
2. Train model (`main_train.py`)
3. Launch API (`main_app_launcher.py`)
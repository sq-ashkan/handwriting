# Handwritten Character Recognition (OCR) Project

## Main Goal
Develop a system capable of recognizing handwritten uppdercase characters (letters or numbers) uploaded by a user with **over 99% accuracy** in demo and real-user testing environments.

## Development Environment Specifications
- **Operating System**: macOS
- **Processor**: Apple M2 Ultra (24-core CPU, 76-core GPU)
- **RAM**: Minimum 40GB free
- **Priority**: High accuracy in real-world conditions
- **Coding Language**: English
- **Documentation Language**: English

## Project Structure
```
├── main_orchestrator.py                 # all workflos manager
├── main_modifier.py                     # split and make data ready           from final to processed / cleaner
├── main_data_enhancers.py               # Main enhancement orchestrator       from temp to final
├── main_processor.py                    # Main data processor                 from raw to temp
├── main_download.py                     # Main data downloader                download raw
├── analyse.py                           # Dataset analyzer
├── requirements.txt                     # Project dependencies
├── src/                                 # Core project code
│   ├── services/                        # Main services
│   │   ├── enhancers/                   # Enhancement modules
│   │   │   ├── base_enhancer.py        # Base enhancement class
│   │   │   ├── brightness_enhancer.py   # Brightness normalization
│   │   │   ├── noise_enhancer.py       # Noise reduction
│   │   │   ├── stroke_enhancer.py      # Stroke width normalization
│   │   │   ├── quality_enhancer.py     # Quality improvement
│   │   │   └── data_splitter.py        # Dataset splitting
│   │   ├── downloaders/                 # Dataset downloaders
│   │   │   ├── english_handwritten.py   # English dataset downloader ✓
│   │   │   ├── mnist_downloader.py      # MNIST downloader ✓
│   │   │   ├── az_downloader.py         # A-Z downloader ✓
│   │   │   └── chars74k_downloader.py   # Chars74K downloader ✓
│   │   ├── processors/                  # Data processors
│   │   │   ├── english_handwritten.py
│   │   │   ├── mnist_processor.py
│   │   │   ├── az_processor.py
│   │   │   └── chars74k_processor.py
│   │   ├── preprocessor/                # Image pre-processing
│   │   └── trainer/                     # Model training system
│   ├── models/                          # Model architecture
│   │   ├── layers.py                    # Network layers
│   │   └── network.py                   # Network configuration
│   └── lib/                             # Utility libraries
│       ├── utils.py                     # General functions
│       ├── cache_manager.py             # Cache and temporary file management
│       └── constants.py                 # Constant variables
└── data/                                # Datasets
    └── processed/                       # Processed data
        ├── digits/                      # Numeric characters (0-9)
        │   ├── 0/                       # Character folder
        │   │   ├── documentation.txt    # Image-label mappings
        │   │   └── images/             # PNG image files
        │   ├── 1/
        │   │   ├── documentation.txt
        │   │   └── images/
        │   └── ...
        └── uppercase/                   # Uppercase letters (A-Z)
            ├── A/                       # Character folder
            │   ├── documentation.txt    # Image-label mappings
            │   └── images/             # PNG image files
            ├── B/
            │   ├── documentation.txt
            │   └── images/
            └── ...
```
## Processing Flow
1. Raw data → Downloaders
2. Downloaded data → Processors
3. Processed data → Enhancers:
   - Brightness normalization
   - Noise reduction
   - Stroke width normalization
   - Quality improvement
4. Enhanced data → Data splitting (train/val/test)

## Technical Notes
- All enhancement modules use parallel processing
- Optimized for M2 Ultra architecture
- Strong error handling and logging
- Cache management for temporary files
- Each module follows SOLID principles
- all images 27x27 are PNG
- no Class based files, just functional and clean code with seperate layer solid
- Flask should be added on trained system to provide webservice to revieve PNG image and recognise the letter or number
- test with the postman
- user part should be done with next.js and user can upload and test the system with the image with mobile

## Data Set report
{
    "digits": {
        "9": 20000,
        "0": 20000,
        "7": 20000,
        "6": 20000,
        "1": 20000,
        "8": 20000,
        "4": 20000,
        "3": 20000,
        "2": 20000,
        "5": 20000
    },
    "uppercase": {
        "R": 20000,
        "U": 20000,
        "I": 20000,
        "N": 20000,
        "G": 20000,
        "Z": 20000,
        "T": 20000,
        "S": 20000,
        "A": 20000,
        "F": 20000,
        "O": 20000,
        "H": 20000,
        "M": 20000,
        "J": 20000,
        "C": 20000,
        "D": 20000,
        "V": 20000,
        "Q": 20000,
        "X": 20000,
        "E": 20000,
        "B": 20000,
        "K": 20000,
        "L": 20000,
        "Y": 20000,
        "P": 20000,
        "W": 20000
    }
}
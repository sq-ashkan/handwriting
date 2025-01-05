# Handwritten Character Recognition (OCR) Project

## Main Goal
Develop a system capable of recognizing handwritten characters (letters or numbers) uploaded by a user with **over 99% accuracy** in demo and real-user testing environments.

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
        ├── lowercase/                   # Lowercase letters (a-z)
        │   ├── a/                       # Character folder
        │   │   ├── documentation.txt    # Image-label mappings
        │   │   └── images/             # PNG image files
        │   ├── b/
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

## Documentation Format
Each character folder contains a documentation.txt file that maps image files to their corresponding labels. The format follows this structure:

```
Ash_PNG_000002 0
Ash_PNG_000022 0
Ash_PNG_000035 0
```

Where:
- The first column (e.g., Ash_PNG_000002) is a unique identifier across the entire dataset
- The second column represents the character label for that image
- Each image identifier is guaranteed to be unique across all character categories

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
- More than 500k png images 27x27 
- no Class based files, just functional and clean code with seperate layer solid
- Flask should be added on trained system
- test with the postman
- user part should be done with next.js and user can upload and test the system with the image with mobile

## Data Set report
{
    "digits": {
        "9": 7974,
        "0": 7919,
        "7": 8309,
        "6": 7892,
        "1": 8893,
        "8": 7841,
        "4": 7840,
        "3": 8157,
        "2": 8006,
        "5": 7329
    },
    "lowercase": {
        "r": 1016,
        "u": 1016,
        "i": 1016,
        "n": 1016,
        "g": 1016,
        "z": 1016,
        "t": 1016,
        "s": 1016,
        "a": 1016,
        "f": 1016,
        "o": 1016,
        "h": 1016,
        "m": 1016,
        "j": 1016,
        "c": 1016,
        "d": 1016,
        "v": 1016,
        "q": 1016,
        "x": 1016,
        "e": 1016,
        "b": 1016,
        "k": 1016,
        "l": 1016,
        "y": 1016,
        "p": 1016,
        "w": 1016
    },
    "uppercase": {
        "R": 12582,
        "U": 30024,
        "I": 2136,
        "N": 20026,
        "G": 6778,
        "Z": 7092,
        "T": 23511,
        "S": 49435,
        "A": 14885,
        "F": 2179,
        "O": 58841,
        "H": 8234,
        "M": 13352,
        "J": 9509,
        "C": 24425,
        "D": 11150,
        "V": 5198,
        "Q": 6828,
        "X": 7288,
        "E": 12456,
        "B": 9684,
        "K": 6619,
        "L": 12602,
        "Y": 11875,
        "P": 20357,
        "W": 11800
    }
}
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
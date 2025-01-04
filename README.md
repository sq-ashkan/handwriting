# Handwritten Character Recognition (OCR) Project

## Main Goal
Develop a system capable of recognizing handwritten characters (letters or numbers) uploaded by a user with **over 99% accuracy** in demo and real-user testing environments, primarily for presentation to investors.

## Development Environment Specifications
- **Operating System**: macOS
- **Processor**: Apple M2 Ultra (24-core CPU, 76-core GPU)
- **RAM**: Minimum 40GB free
- **Priority**: High accuracy in real-world conditions
- **Coding Language**: English
- **Documentation Language**: Persian/English

## Project Structure
```
/Users/roammer/Documents/Github/handwriting/
├── main_data_enhancers.py               # Main enhancement orchestrator
├── main_download.py                     # Main data downloader
├── main_processor.py                    # Main data processor
├── analyse.py                           # Dataset analyzer
├── requirements.txt                     # Project dependencies
├── src/                                 # Core project code
│   ├── services/                        # Main services
│   │   ├── enhancers/                   # Enhancement modules
│   │   │   ├── __init__.py
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
├── data/                                # Datasets
│   ├── raw/                             # Raw data
│   │   ├── english_handwritten/         # English Handwritten dataset
│   │   │   ├── images/                  # Standardized images
│   │   │   └── documentation.txt        # Standard format documentation
│   │   ├── mnist/                       # MNIST dataset
│   │   │   ├── images/                  # Standardized images
│   │   │   └── documentation.txt        # Standard format documentation
│   │   ├── az_handwritten/              # A-Z dataset
│   │   │   ├── images/                  # Standardized images
│   │   │   └── documentation.txt        # Standard format documentation
│   │   └── chars74k/                    # Chars74K dataset
│   │       ├── images/                  # 27x27 pixel images
│   │       └── documentation.txt        # Standard format documentation
│   ├── processed/                       # Processed data
│   ├── temp/                            # Temporary data
│   │   ├── EH/                          # Temporary English Handwritten data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── MNIST/                       # Temporary MNIST data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── AZ/                          # Temporary A-Z data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   └── Chars74K/                    # Temporary Chars74K data
│   │       ├── images/    
│   │       └── documentation.txt
│   └── logs/                            # Logs and reports
└── tests/                               # Unit tests
    ├── test_downloaders/
    ├── test_preprocessor/
    └── test_trainer/
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
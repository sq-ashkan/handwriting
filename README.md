
# Handwritten Character Recognition (OCR) Project

## Main Goal
Develop a system capable of recognizing handwritten characters (letters or numbers) uploaded by a user with **over 99% accuracy** in demo and real-user testing environments, primarily for presentation to investors.

## Development Environment Specifications
- **Operating System**: macOS
- **Processor**: Apple M2 Ultra
- **Priority**: High accuracy in real-world conditions
- **Coding Language**: English
- **Documentation Language**: Persian

## Updated Project Structure
```
/Users/roammer/Documents/Github/handwriting/
├── src/                                # Core project code
│   ├── services/                       # Main services
│   │   ├── downloaders/                # Dataset downloaders
│   │   │   ├── english_handwritten.py  # English Handwritten Downloader ✓
│   │   │   ├── mnist_downloader.py     # MNIST Downloader ✓
│   │   │   ├── az_downloader.py        # A-Z Downloader ✓
│   │   │   └── chars74k_downloader.py  # Chars74K Downloader ✓
│   │   ├── processors/                 # Data processors
│   │   │   ├── english_handwritten.py
│   │   │   ├── mnist_processor.py
│   │   │   ├── az_processor.py
│   │   │   └── chars74k_processor.py
│   │   ├── preprocessor/               # Image pre-processing
│   │   └── trainer/                    # Model training system
│   ├── models/                         # Model architecture
│   │   ├── layers.py                   # Network layers
│   │   └── network.py                  # Network configuration
│   └── lib/                            # Utility libraries
│       ├── utils.py                    # General functions
│       ├── cache_manager.py            # Cache and temporary file management
│       └── constants.py                # Constant variables
├── data/                               # Datasets
│   ├── raw/                            # Raw data
│   │   ├── english_handwritten/        # English Handwritten dataset
│   │   │   ├── images/                 # Standardized images
│   │   │   └── documentation.txt       # Standard format documentation
│   │   ├── mnist/                      # MNIST dataset
│   │   │   ├── images/                 # Standardized images
│   │   │   └── documentation.txt       # Standard format documentation
│   │   ├── az_handwritten/             # A-Z dataset
│   │   │   ├── images/                 # Standardized images
│   │   │   └── documentation.txt       # Standard format documentation
│   │   └── chars74k/                   # Chars74K dataset
│   │       ├── images/                 # 27x27 pixel images
│   │       └── documentation.txt       # Standard format documentation
│   ├── processed/                      # Processed data
│   ├── temp/                           # Temporary data
│   │   ├── EH/                         # Temporary English Handwritten data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── MNIST/                      # Temporary MNIST data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   ├── AZ/                         # Temporary A-Z data
│   │   │   ├── images/    
│   │   │   └── documentation.txt
│   │   └── Chars74K/                   # Temporary Chars74K data
│   │       ├── images/    
│   │       └── documentation.txt
│   └── logs/                           # Logs and reports
├── tests/                              # Unit tests
│   ├── test_downloaders/
│   ├── test_preprocessor/
│   └── test_trainer/
└── requirements.txt                    # Dependencies
```

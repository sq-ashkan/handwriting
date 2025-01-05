# Handwritten Character Recognition (OCR) Project

## Main Goal
Develop a system capable of recognizing handwritten uppdercase characters (letters or numbers) uploaded by a user with **over 99% accuracy** in demo and real-user testing environments.

## Development Environment Specifications
- **Operating System**: macOS
- **Processor**: Apple M2 Ultra (24-core CPU, 76-core GPU)
- **RAM**: Minimum 40GB free
- **Priority**: High accuracy in real-world conditions

## Project Structure
```
├── api/
    └── inference.py                     # serverless for Vercel Server
├── main_orchestrator.py                 # all workflows manager
├── best_model.pth                       # final model from
├── requirements.txt                     # Project dependencies
└── src/                                # Core project code
    ├── api/                            # API related files
    │   └── api.operator.py            # API operations handler
    ├── lib/                            # Utility libraries
    │   ├── cache_manager.py           # Cache management
    │   ├── config.py                  # Configuration settings
    │   ├── constants.py               # Constant variables
    │   └── utils.py                   # Utility functions
    ├── model/                          # Model related files
    │   └── train.operator.py          # Training operations
    ├── operators/                      # Main operators
    │   ├── main_data_enhancers.operator.py
    │   ├── main_downloader.operator.py
    │   ├── main_modifier.operator.py
    │   └── main_processor.operator.py
    └── services/                       # Core services
        ├── downloaders/                # Dataset downloaders
        │   ├── az_downloader.py
        │   ├── chars74k_downloader.py
        │   ├── english_handwritten.py
        │   └── mnist_downloader.py
        ├── enhancers/                  # Enhancement modules
        │   ├── base_enhancer.py
        │   ├── brightness_enhancer.py
        │   ├── data_splitter.py
        │   ├── noise_enhancer.py
        │   ├── quality_enhancer.py
        │   └── stroke_enhancer.py
        ├── mergers/                    # Data merging modules
        │   ├── cache_cleaner.py
        │   ├── cleaner.py
        │   ├── merger.py
        │   ├── splitter.py
        │   └── verifier.py
        ├── preprocessor/               # Image pre-processing
        └── processors/                 # Data processors
            ├── az_processor.py
            ├── chars74k_processor.py
            ├── english_handwritten.py
            └── mnist_processor.py

└── data/                               # Datasets
    └── processed/                      # Processed data
        ├── digits/                     # Numeric characters (0-9)
        │   ├── 0/                      # Character folder
        │   │   ├── documentation.txt   # Image-label mappings
        │   │   └── images/            # PNG image files
        │   └── ...
        └── uppercase/                  # Uppercase letters (A-Z)
            ├── A/                      # Character folder
            │   ├── documentation.txt   # Image-label mappings
            │   └── images/            # PNG image files
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
- Flask for api
- test with the postman
- user part with next.js/react.js

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
# Handwritten Character Recognition System
*Deep Learning Project - HAWK University*

## Author
**Ashkan Sadri Ghamshi**  
HAWK University  
Jan 2025

## Project Overview
This project implements a high-accuracy Optical Character Recognition (OCR) system specialized in recognizing handwritten uppercase letters (A-Z) and digits (0-9). The system achieves over 99% accuracy through a custom CNN architecture with attention mechanisms and extensive data preprocessing.

## Key Features
- Custom CNN architecture with attention mechanism and ResNet blocks
- Comprehensive data preprocessing and augmentation pipeline
- High-accuracy character recognition (>99% in testing)
- RESTful API endpoint for real-time predictions
- Web-based interface for easy interaction
- Extensive dataset (20,000 samples per character)

## Technical Architecture

### Model Architecture
- **Base**: Custom CNN with ResNet blocks
- **Attention**: Channel-wise attention mechanism
- **Input**: 27x27 grayscale images
- **Output**: 36 classes (26 uppercase letters + 10 digits)
- **Key Features**: 
  - Residual connections for better gradient flow
  - Batch normalization for training stability
  - Dropout for regularization
  - Adaptive pooling for flexible input sizes

### Project Structure
```
├── main_orchestrator.py                              # Main workflow manager
├── best_model.pth                                    # Trained model weights
├── requirements.txt                                  # Project dependencies
└── src/                                              # Core project code
    ├── lib/                                          # Utility libraries
        ├── cache_manager.py/                         # cleaning cashe
        ├── config.py/                                # Config variables
        ├── constans.py/                              # Settings varialbles
        └── utils.py/                                 # utils - setup loging system
    ├── operators/                                  
        ├── main_downloader.operator.py               # (Step 1) Download 4 Datasets: Mnist - EH - A-Z - Chars74K
        ├── main_processor.operator.py                # (Step 2) first step of proccessing like size and remove garbage data
        ├── main_data_enhancers.operator.py           # (Step 3) work on general quality
        ├── main_modifier.operator.py                 # (Step 4) Merg and verify all datasets together
        └── main_augmentation.operator.py             # (Step 5) Data augmentator make Data ready
    ├── model                                     
            └── train.operator.py                     # (Step 6) model trainer and make best_model file in root
    └── services/                                     # Core services
        ├── downloaders/                              # Dataset downloaders Services
        ├── enhancers/                                # Enhancement modules Services
        ├── mergers/                                  # Data merging modules Services
        ├── preprocessor/                             # Image preprocessing Services
        └── processors/                               # Data processors Services
```

## Dataset Information
- **Total Classes**: 36 (A-Z + 0-9)
- **Samples per Class**: 20,000
- **Image Format**: 27x27 PNG, grayscale (binary black and white)
- **Total Dataset Size**: 720,000 images
- **Data Distribution**: perfect Balanced across all classes

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8.1+
- CUDA capable GPU (for training) / CPU (for inference)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sq-ashkan/handwriting 
   cd handwriting
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n handwriting python=3.8
   conda activate handwriting
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Backend API
1. Start the Flask server:
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5001`

2. Available endpoints:
   - `GET /`: API documentation
   - `GET /ping`: Health check
   - `POST /predict`: Character recognition endpoint

#### Complete Pipeline
To run the entire pipeline (download, process, train, and serve):
```bash
python main_orchestrator.py
```

## API Usage

### Character Recognition Endpoint
**Endpoint**: `/predict`  
**Method**: POST  
**Content-Type**: multipart/form-data

**Example Request**:
```python
import requests

files = {'image': open('character.png', 'rb')}
response = requests.post('http://localhost:5001/predict', files=files)
print(response.json())
```

**Example Response**:
```json
{
    "character": "A",
    "confidence": 0.998,
    "preprocessed_image": "base64_encoded_string",
    "status": "success"
}
```

## Model Performance
- Training Accuracy: 99.5%
- Validation Accuracy: 99.2%
- Test Accuracy: 99.1%
- Average Inference Time: <50ms

## Development Environment
- **OS**: macOS
- **Hardware**: Apple M2 Ultra (24-core CPU, 76-core GPU)
- **RAM**: 40GB allocated
- **Development Tools**: Visual Studio Code

## Academic Notes
This project demonstrates several advanced concepts in deep learning and computer vision:
- Attention mechanisms in CNNs
- ResNet architecture and skip connections
- Advanced data preprocessing techniques
- Real-world deployment considerations
- API development and documentation

## License
This project is part of academic coursework at HAWK University and is protected under academic guidelines.

## Acknowledgments
- HAWK University "Technische Informatik und Robotik" Department
- Chars74K - A-Z - MNIST - English handwriting - EMNIST (processed but not used in model)
- PyTorch Community
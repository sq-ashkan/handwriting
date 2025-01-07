"""
Handwritten Character Recognition API
-----------------------------------

Author: Ashkan Sadri Ghamshi
Project: Advanced OCR System for Handwritten Character Recognition
Version: 1.0.0
License: Proprietary
Created: January 2024
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.inference import OCRPredictor
import uvicorn
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Handwriting OCR API",
    description="API for handwritten character recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR predictor
predictor = OCRPredictor()

async def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check if file is not empty
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file size (max 1MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    if file_size > 1_000_000:  # 1MB
        raise HTTPException(status_code=400, detail="File size too large. Maximum size is 1MB")
    
    # Validate content type
    content_type = file.content_type
    if not content_type or not content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only image files are allowed"
        )

@app.get("/", response_model=Dict[str, Any])
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "status": "active",
        "message": "Handwriting OCR API is running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "OCR API",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service health check failed")

@app.post("/predict", response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict character from uploaded image.
    
    Parameters:
        file (UploadFile): Image file to analyze
        
    Returns:
        dict: Prediction results including character and confidence
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Validate file
        await validate_file(file)
        
        # Get prediction
        result = await predictor.predict_character(file)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        # Add processing time to result
        result["processing_time"] = processing_time
        
        # Log successful prediction
        logger.info(f"Successfully predicted character: {result['character']} "
                   f"with confidence: {result['confidence']:.4f}")
        
        return JSONResponse(content=result)
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # Log error
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "code": exc.status_code
        }
    )

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
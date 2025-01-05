from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.inference import OCRPredictor
import uvicorn
from fastapi.responses import JSONResponse
import time

app = FastAPI(
    title="Handwriting OCR API",
    description="API for handwritten character recognition",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR predictor
predictor = OCRPredictor()

@app.get("/health")
async def health_check():
    """Health check endpoint for Koyeb"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Handwriting OCR API is running",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the character in the uploaded image
    """
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Get prediction
        result = await predictor.predict_character(file)
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 3)
        
        return JSONResponse({
            "status": "success",
            "prediction": result["character"],
            "confidence": round(float(result["confidence"]), 4),
            "processing_time": processing_time
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

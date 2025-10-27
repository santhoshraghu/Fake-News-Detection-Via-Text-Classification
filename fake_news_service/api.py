from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
from .utils import load_model, predict_text


# Global variables for model and vectorizer
model = None
vectorizer = None

@asynccontextmanager
async def startup_and_shutdown(app: FastAPI):
    # Startup
    global model, vectorizer
    
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise HTTPException(
            status_code=500,
            detail="Model files not found. Please train the model first using scripts/train.py"
        )
    
    try:
        model, vectorizer = load_model(model_path, vectorizer_path)
        print("Model and vectorizer loaded successfully!")
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )
    
    yield  # App runs
    
    # Shutdown
    model = None
    vectorizer = None

app = FastAPI(
    title="Fake News Detection Service",
    description="An Initial Implementation for detecting fake news using text classification",
    version="1.0.0",
    lifespan=startup_and_shutdown
)

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    text: str


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str

@app.get("/")
async def root():
    """A simple root endpoint."""
    return {
        "message": "Fake News Detection Service", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to check if model and vectorizer are loaded."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts if the given text is fake or real news.
    
    Args:
        request: Prediction request containing text
        
    Returns:
        Prediction result ( FAKE or REAL)
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model or Vectorizer not loaded. Please check logs."
        )
    
    try:
        prediction = predict_text(model, vectorizer, request.text)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fake_news_service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
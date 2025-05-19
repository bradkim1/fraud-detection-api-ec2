#!/usr/bin/env python3

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

# Add the current directory to the Python path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Create model directory if it doesn't exist
model_dir = os.path.join(current_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

# Try to import the prediction module
try:
    sys.path.append(model_dir)
    from pipeline_module import FraudDetectionPipeline
    pipeline = FraudDetectionPipeline()
    model_loaded = True
    print("Successfully loaded fraud detection model")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

app = FastAPI(title="Fraud Detection API")

class TransactionRequest(BaseModel):
    data: Dict[str, Any]

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first using train_model.py"
        )
        
    try:
        prediction, probability = pipeline.predict(request.data)
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    if model_loaded:
        return {"message": "Fraud Detection API is running. Use /predict endpoint."}
    else:
        return {
            "message": "Fraud Detection API is running but model is not loaded.",
            "instructions": "Please train the model first using train_model.py"
        }

@app.get("/status")
async def status():
    """Return status information about the API and model"""
    return {
        "model_loaded": model_loaded,
        "base_directory": current_dir,
        "model_directory": model_dir,
        "model_files": os.listdir(model_dir) if os.path.exists(model_dir) else []
    }

if __name__ == "__main__":
    print(f"Starting Fraud Detection API on port 8001")
    uvicorn.run("api:app", host="0.0.0.0", port=8001)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define request and response models
class TransactionRequest(BaseModel):
    data: Dict[str, Any]

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool

# Define variables for model and pipeline
model = None
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Load model and pipeline when the API starts"""
    global model, pipeline
    
    MODEL_DIR = '/home/ubuntu/model-ec2/model'
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
    
    logger.info("Loading model and pipeline...")
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    
    try:
        pipeline = joblib.load(pipeline_path)
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        pipeline = None

@app.get("/")
async def root():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    pipeline_status = "loaded" if pipeline is not None else "not loaded"
    return {
        "message": "Fraud Detection API is running.",
        "model_status": model_status,
        "pipeline_status": pipeline_status
    }

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    """Make fraud predictions"""
    global model, pipeline
    
    # Check if model and pipeline are loaded
    if model is None or pipeline is None:
        # Try loading again if they're not loaded
        MODEL_DIR = '/home/ubuntu/model-ec2/model'
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
        
        try:
            if model is None:
                model = joblib.load(model_path)
            if pipeline is None:
                pipeline = joblib.load(pipeline_path)
        except Exception as e:
            logger.error(f"Error loading model or pipeline: {e}")
            raise HTTPException(status_code=503, detail="Model or pipeline not loaded")
    
    try:
        # Log the received data
        logger.info(f"Received prediction request with data keys: {list(request.data.keys())}")
        
        # Convert dict to DataFrame
        data = pd.DataFrame([request.data])
        logger.info(f"Created DataFrame with shape: {data.shape}")
        
        # Try to transform the data
        try:
            # Check if the pipeline has the transform_for_predict method
            if hasattr(pipeline, 'transform_for_predict'):
                X = pipeline.transform_for_predict(data)
                logger.info(f"Transformed data shape: {X.shape}")
            else:
                # If not, try to use pipeline directly
                X = pipeline.transform(data)
                logger.info(f"Transformed data shape: {X.shape}")
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise HTTPException(status_code=400, detail=f"Error transforming data: {str(e)}")
        
        # Make prediction
        try:
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0, 1])
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            
            return {
                "prediction": prediction,
                "probability": probability,
                "is_fraud": bool(prediction == 1)
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Fraud Detection API on port 8007...")
    uvicorn.run("improved_fraud_api:app", host="0.0.0.0", port=8007)

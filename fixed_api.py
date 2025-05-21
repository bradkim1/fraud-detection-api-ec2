from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
import os
import uvicorn
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("fraud_api")

# Define the app
app = FastAPI(title="Fixed Fraud Detection API")

# Global variables
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = None
request_count = 0
prediction_count = 0
error_count = 0
fraud_count = 0
uptime_start = time.time()

# Load model features
try:
    with open(os.path.join(os.path.dirname(__file__), "model_features.json"), "r") as f:
        MODEL_FEATURES = json.load(f)
    logger.info(f"Loaded {len(MODEL_FEATURES)} feature names from model_features.json")
except Exception as e:
    # Basic fallback features
    MODEL_FEATURES = [
        "TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "C1", "C2", "D1", "D2"
    ]
    logger.warning(f"Error loading model_features.json: {e}. Using fallback features.")

# Load model
try:
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully: {type(model).__name__}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# Define the models
@app.post("/predict")
async def predict(request: Request):
    """Make fraud predictions with flexible input format"""
    global model, request_count, prediction_count, error_count, fraud_count
    
    # Generate request ID
    with request_count_lock:
        global request_count
        request_count += 1
        current_count = request_count
    
    request_id = f"pred-{int(time.time())}-{current_count}"
    
    # Record start time
    start_process = time.time()
    
    # Check if model is loaded
    if model is None:
        with error_count_lock:
            error_count += 1
        logger.error(f"Request {request_id}: Model not loaded")
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    
    try:
        # Parse request body
        body_bytes = await request.body()
        body_str = body_bytes.decode()
        logger.info(f"Request {request_id}: Raw request: {body_str}")
        
        # Try to parse as JSON
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError as e:
            logger.error(f"Request {request_id}: JSON decode error: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {str(e)}"})
        
        # Extract transaction data
        transaction_data = {}
        transaction_id = "unknown"
        
        if isinstance(body, dict):
            # Try different possible formats
            if "data" in body and isinstance(body["data"], dict):
                transaction_data = body["data"]
                if "transaction_id" in body:
                    transaction_id = body["transaction_id"]
            elif "TransactionAmt" in body or any(key.startswith(('C', 'D', 'V')) for key in body.keys()):
                # Body itself appears to contain transaction data
                transaction_data = body
                if "transaction_id" in body:
                    transaction_id = body["transaction_id"]
            else:
                # Treat all fields as transaction data
                transaction_data = body
        
        logger.info(f"Request {request_id}: Extracted transaction data: {transaction_data}")
        
        if not transaction_data:
            logger.error(f"Request {request_id}: No transaction data found in request")
            return JSONResponse(status_code=422, content={"detail": "No transaction data found in request"})
        
        # Create DataFrame with all required features
        data = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)
        
        # Fill in values from transaction data where available
        for col in MODEL_FEATURES:
            if col in transaction_data:
                data[col] = transaction_data[col]
        
        # Map Amount to TransactionAmt if needed
        if 'Amount' in transaction_data and 'TransactionAmt' in MODEL_FEATURES:
            data['TransactionAmt'] = transaction_data['Amount']
        
        # Add basic calculated features
        # C_sum
        c_cols = [col for col in data.columns if col.startswith('C') and col != 'C_sum']
        if 'C_sum' in MODEL_FEATURES and c_cols:
            data['C_sum'] = data[c_cols].sum(axis=1)
        
        # D_missing
        d_cols = [col for col in data.columns if col.startswith('D')]
        if 'D_missing' in MODEL_FEATURES and d_cols:
            data['D_missing'] = data[d_cols].isna().sum(axis=1)
        
        # Hour and day
        current_datetime = datetime.now()
        if 'hour' in MODEL_FEATURES:
            data['hour'] = current_datetime.hour
        if 'day' in MODEL_FEATURES:
            data['day'] = current_datetime.weekday()
        
        logger.info(f"Request {request_id}: Data prepared for prediction, shape: {data.shape}")
        
        # Make prediction
        try:
            prediction = int(model.predict(data)[0])
            probability = float(model.predict_proba(data)[0, 1])
            
            with prediction_count_lock:
                prediction_count += 1
            
            # Increment fraud count if fraud detected
            if prediction == 1:
                with fraud_count_lock:
                    fraud_count += 1
                
            processing_time = time.time() - start_process
            
            # Prepare result
            result = {
                "transaction_id": transaction_id,
                "prediction": prediction,
                "probability": probability,
                "is_fraud": bool(prediction == 1),
                "request_id": request_id,
                "processing_time": processing_time
            }
            
            logger.info(f"Request {request_id}: Prediction: {prediction}, Probability: {probability:.4f}, Time: {processing_time:.4f}s")
            
            return result
        except Exception as e:
            with error_count_lock:
                error_count += 1
            logger.error(f"Request {request_id}: Error making prediction: {e}")
            return JSONResponse(status_code=400, content={"detail": f"Error making prediction: {str(e)}"})
            
    except Exception as e:
        with error_count_lock:
            error_count += 1
        logger.error(f"Request {request_id}: Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

class TransactionRequest(BaseModel):
    transaction_id: str
    data: Dict[str, Any] = Field(..., description="Transaction data with features")

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool
    request_id: str
    processing_time: float

# Locks for thread safety
request_count_lock = threading.Lock()
prediction_count_lock = threading.Lock()
error_count_lock = threading.Lock()
fraud_count_lock = threading.Lock()

# Root endpoint
@app.get("/")
def read_root():
    """API status endpoint"""
    uptime = time.time() - uptime_start
    return {
        "message": "Enhanced Fraud Detection API is running.",
        "model_status": "loaded" if model is not None else "not loaded",
        "uptime": f"{uptime:.2f} seconds",
        "request_count": request_count,
        "prediction_count": prediction_count
    }

# Predict endpoint
@app.post("/predict", 
    response_model=TransactionResponse,
    summary="Fraud Prediction",
    description="Analyzes a transaction and returns the probability of it being fraudulent",
    response_description="Prediction results including fraud probability"
)
async def predict(request: TransactionRequest):
    """Make fraud predictions"""
    global model, request_count, prediction_count, error_count, fraud_count
    
    # Generate request ID
    with request_count_lock:
        global request_count
        request_count += 1
        current_count = request_count
    
    request_id = f"pred-{int(time.time())}-{current_count}"
    
    # Record start time
    start_process = time.time()
    
    # Check if model is loaded
    if model is None:
        with error_count_lock:
            error_count += 1
        logger.error(f"Request {request_id}: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Log the received data
        transaction_data = request.data
        logger.info(f"Request {request_id}: Received prediction request with {len(transaction_data)} features")
        
        # Create DataFrame with all required features
        data = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)
        
        # Fill in values from transaction data where available
        for col in MODEL_FEATURES:
            if col in transaction_data:
                data[col] = transaction_data[col]
        
        # Map Amount to TransactionAmt if needed
        if 'Amount' in transaction_data and 'TransactionAmt' in MODEL_FEATURES:
            data['TransactionAmt'] = transaction_data['Amount']
        
        # Add basic calculated features
        # C_sum
        c_cols = [col for col in data.columns if col.startswith('C') and col != 'C_sum']
        if 'C_sum' in MODEL_FEATURES and c_cols:
            data['C_sum'] = data[c_cols].sum(axis=1)
        
        # D_missing
        d_cols = [col for col in data.columns if col.startswith('D')]
        if 'D_missing' in MODEL_FEATURES and d_cols:
            data['D_missing'] = data[d_cols].isna().sum(axis=1)
        
        # Hour and day
        current_datetime = datetime.now()
        if 'hour' in MODEL_FEATURES:
            data['hour'] = current_datetime.hour
        if 'day' in MODEL_FEATURES:
            data['day'] = current_datetime.weekday()
        
        logger.info(f"Request {request_id}: Data prepared for prediction, shape: {data.shape}")
        
        # Make prediction
        try:
            prediction = int(model.predict(data)[0])
            probability = float(model.predict_proba(data)[0, 1])
            
            with prediction_count_lock:
                prediction_count += 1
            
            # Increment fraud count if fraud detected
            if prediction == 1:
                with fraud_count_lock:
                    fraud_count += 1
                
            processing_time = time.time() - start_process
            
            # Prepare result
            result = {
                "prediction": prediction,
                "probability": probability,
                "is_fraud": bool(prediction == 1),
                "request_id": request_id,
                "processing_time": processing_time
            }
            
            logger.info(f"Request {request_id}: Prediction: {prediction}, Probability: {probability:.4f}, Time: {processing_time:.4f}s")
            
            return result
        except Exception as e:
            with error_count_lock:
                error_count += 1
            logger.error(f"Request {request_id}: Error making prediction: {e}")
            raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        with error_count_lock:
            error_count += 1
        logger.error(f"Request {request_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint
@app.post("/debug")
async def debug(request: TransactionRequest):
    """Debug endpoint for testing."""
    try:
        # Extract transaction data
        transaction_data = request.data
        
        # Create DataFrame with all required features
        data = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES)
        
        # Fill in values from transaction data where available
        for col in MODEL_FEATURES:
            if col in transaction_data:
                data[col] = transaction_data[col]
        
        return {
            "status": "success",
            "original_features": list(transaction_data.keys()),
            "model_features": MODEL_FEATURES[:10] + ["..."] + MODEL_FEATURES[-10:],
            "model_feature_count": len(MODEL_FEATURES),
            "dataframe_shape": str(data.shape),
            "model_type": str(type(model))
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8015)

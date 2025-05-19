from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os

# Add the model directory to the Python path
model_dir = '/home/ubuntu/model-ec2/model'
sys.path.append(model_dir)

# Import the FraudDetectionPipeline class
try:
    from pipeline_module import FraudDetectionPipeline
    # Initialize the pipeline
    fraud_pipeline = FraudDetectionPipeline()
    print("Fraud Detection Pipeline loaded successfully")
except Exception as e:
    print(f"Error loading Fraud Detection Pipeline: {e}")
    fraud_pipeline = None

app = FastAPI(title="Fraud Detection API")

class TransactionRequest(BaseModel):
    data: Dict[str, Any]

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    if fraud_pipeline is None:
        raise HTTPException(status_code=503, detail="Fraud Detection Pipeline not loaded")
    
    try:
        # Use the pipeline to predict
        prediction, probability = fraud_pipeline.predict(request.data)
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    pipeline_status = "loaded" if fraud_pipeline is not None else "not loaded"
    return {
        "message": f"Fraud Detection API is running. Pipeline status: {pipeline_status}. Use /predict endpoint."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")

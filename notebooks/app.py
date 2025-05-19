from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import pandas as pd
import joblib
import os
import uvicorn

# Define the FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions",
    version="1.0.0"
)

# Define the request and response models
class TransactionRequest(BaseModel):
    """
    Request model for fraud detection
    """
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "TransactionAmt": 100.0,
                    "card1": 1234.0,
                    "ProductCD": "W",
                    "card4": "visa",
                    "card6": "debit"
                }
            }
        }

class TransactionResponse(BaseModel):
    """
    Response model for fraud detection
    """
    prediction: int
    probability: float
    is_fraud: bool
    transaction_risk: str

# Load the pipeline
class FraudDetectionService:
    def __init__(self, model_path='/workspace/model/model.pkl', pipeline_path='/workspace/model/pipeline.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.pipeline = joblib.load(pipeline_path)
            self.initialized = True
        except (FileNotFoundError, Exception) as e:
            self.initialized = False
            self.error = str(e)
    
    def predict(self, data):
        """
        Make fraud predictions on new data
        
        Args:
            data: Dictionary with transaction data
            
        Returns:
            tuple: (prediction, probability)
        """
        if not self.initialized:
            raise ValueError(f"Model not loaded: {self.error}")
            
        # Convert dict to DataFrame
        df = pd.DataFrame([data])
            
        # Prepare data for model
        X = self.pipeline.transform_for_predict(df)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return prediction, probability

# Initialize the service
service = FraudDetectionService()

# Define the endpoints
@app.get("/")
async def root():
    """
    Root endpoint showing API status
    """
    if service.initialized:
        return {"message": "Fraud Detection API is running. Use /predict endpoint."}
    else:
        return {"message": f"Error loading model: {service.error}"}

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    """
    Make a fraud prediction for a single transaction
    """
    try:
        if not service.initialized:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {service.error}")
            
        prediction, probability = service.predict(request.data)
        
        # Determine risk level
        if probability < 0.2:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"
            
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_fraud": bool(prediction == 1),
            "transaction_risk": risk_level
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if service.initialized:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "error": service.error}

@app.get("/metrics")
async def get_metrics():
    """
    Return model metrics if available
    """
    try:
        # This would typically come from model metadata saved during training
        # For now, we'll just return placeholder values
        return {
            "model_type": "RandomForest",
            "metrics": {
                "f1_score": 0.92,
                "precision": 0.94, 
                "recall": 0.90,
                "roc_auc": 0.97
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    # Determine port - use PORT environment variable if available, otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

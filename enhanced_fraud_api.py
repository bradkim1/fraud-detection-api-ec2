from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
import json
from datetime import datetime
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Configure logging
log_dir = "/home/ubuntu/model-ec2/logs"
os.makedirs(log_dir, exist_ok=True)

# Create logger
logger = logging.getLogger("fraud_api")
logger.setLevel(logging.INFO)

# Create file handler for detailed logs
file_handler = logging.FileHandler(f"{log_dir}/fraud_api.log")
file_handler.setLevel(logging.INFO)

# Create console handler for terminal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Create separate logger for API requests
request_logger = logging.getLogger("api_requests")
request_logger.setLevel(logging.INFO)

# Create file handler for request logs
request_file_handler = logging.FileHandler(f"{log_dir}/api_requests.log")
request_file_handler.setLevel(logging.INFO)
request_file_handler.setFormatter(formatter)
request_logger.addHandler(request_file_handler)

# Create separate logger for predictions
prediction_logger = logging.getLogger("predictions")
prediction_logger.setLevel(logging.INFO)

# Create JSON file handler for predictions
prediction_file_handler = logging.FileHandler(f"{log_dir}/predictions.log")
prediction_file_handler.setLevel(logging.INFO)
prediction_logger.addHandler(prediction_file_handler)

# Application metrics
start_time = time.time()
request_count = 0
prediction_count = 0
error_count = 0
fraud_count = 0
response_times = []

# Recreate the AdvancedMLPipeline class
class AdvancedMLPipeline:
    def __init__(self, model_type='rf', n_components=10, remove_outliers=True):
        model_map = {
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.model = model_map.get(model_type)
        self.scaler = StandardScaler()
        self.imputer_num = SimpleImputer(strategy='mean')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.pca = PCA(n_components=n_components)
        self.remove_outliers = remove_outliers
        self.cat_columns = []
        self.feature_columns = []

    def _add_features(self, df):
        # Create time features from TransactionDT
        if 'TransactionDT' in df.columns:
            df['hour'] = pd.to_datetime(df['TransactionDT'], unit='s', errors='coerce').dt.hour.astype('Int8')
            df['day'] = pd.to_datetime(df['TransactionDT'], unit='s', errors='coerce').dt.dayofweek.astype('Int8')
            df.drop(columns=['TransactionDT'], inplace=True)

        # Group-level aggregation features
        c_cols = [col for col in df.columns if col.startswith('C')]
        d_cols = [col for col in df.columns if col.startswith('D')]
        v_cols = [col for col in df.columns if col.startswith('V')]

        if c_cols:
            df['C_sum'] = df[c_cols].sum(axis=1)
        if d_cols:
            df['D_missing'] = df[d_cols].isnull().sum(axis=1)
        if v_cols:
            df['V_mean'] = df[v_cols].mean(axis=1)
        return df

    def _remove_outliers(self, df, col='TransactionAmt'):
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        return df

    def transform_for_predict(self, df):
        df = self._add_features(df)

        # Fill missing categorical values
        for col in self.cat_columns:
            if col not in df.columns:
                df[col] = "missing"
        
        if self.cat_columns:
            df[self.cat_columns] = self.imputer_cat.transform(df[self.cat_columns])

        # Impute numeric
        num_cols = [col for col in df.columns if col not in self.cat_columns + ['TransactionID']]
        if num_cols:
            df[num_cols] = self.imputer_num.transform(df[num_cols])

        # Encode categoricals
        encoded = self.encoder.transform(df[self.cat_columns])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=self.encoder.get_feature_names_out(self.cat_columns), 
            index=df.index
        )

        df = df.drop(columns=self.cat_columns)
        df = pd.concat([df, encoded_df], axis=1)

        # Scale numeric
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        # Apply PCA to V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            pca_trans = self.pca.transform(df[v_cols])
            pca_df = pd.DataFrame(
                pca_trans, 
                columns=[f'V_PCA_{i}' for i in range(pca_trans.shape[1])], 
                index=df.index
            )
            df = df.drop(columns=v_cols)
            df = pd.concat([df, pca_df], axis=1)

        # Align columns with training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        return df

# Create the FastAPI app
app = FastAPI(title="Enhanced Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class TransactionRequest(BaseModel):
    """
    Transaction data for fraud prediction
    """
    data: Dict[str, Any] = Field(..., 
        description="Transaction details including amount, card info, and other features",
        example={
            "TransactionAmt": 100.0,
            "ProductCD": "C",
            "card1": 1234,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com"
        }
    )

class TransactionResponse(BaseModel):
    """
    Fraud prediction response
    """
    prediction: int = Field(..., description="Binary classification (0=legitimate, 1=fraud)")
    probability: float = Field(..., description="Fraud probability between 0-1")
    is_fraud: bool = Field(..., description="Boolean flag indicating fraud detection")
    request_id: str = Field(..., description="Unique identifier for the request")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")

class MetricsResponse(BaseModel):
    uptime: float
    request_count: int
    prediction_count: int
    error_count: int
    fraud_count: int
    avg_response_time: float
    last_5_response_times: list
    model_info: Dict[str, Any]

# Define variables for model and pipeline
model = None
pipeline = None

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    
    # Generate request ID
    request_id = f"req-{int(time.time())}-{request_count}"
    
    # Log the request start
    start_time = time.time()
    request_count += 1
    
    # Get client IP
    client_host = request.client.host if request.client else "unknown"
    
    # Log request details
    request_logger.info(f"Request {request_id} started - Method: {request.method}, Path: {request.url.path}, Client: {client_host}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response_times.append(process_time)
    
    # Keep response_times limited to last 100
    if len(response_times) > 100:
        response_times.pop(0)
    
    # Log response details
    request_logger.info(f"Request {request_id} completed - Status: {response.status_code}, Time: {process_time:.4f}s")
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response

@app.on_event("startup")
async def startup_event():
    """Load model and pipeline when the API starts"""
    global model, pipeline
    
    logger.info("Starting Enhanced Fraud Detection API")
    logger.info(f"Log directory: {log_dir}")
    
    MODEL_DIR = '/home/ubuntu/model-ec2/model'
    model_path = os.path.join(MODEL_DIR, 'model.pkl')
    pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
    
    logger.info("Loading model and pipeline...")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    
    try:
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded successfully: {type(pipeline).__name__}")
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        pipeline = None

@app.get("/", 
    summary="API Health Check",
    description="Returns the status of the API, model, and pipeline",
    response_description="Health status information"
)
async def root():
    """Health check endpoint"""
    global start_time, request_count, prediction_count, error_count
    
    uptime = time.time() - start_time
    
    return {
        "message": "Enhanced Fraud Detection API is running.",
        "model_status": "loaded" if model is not None else "not loaded",
        "pipeline_status": "loaded" if pipeline is not None else "not loaded",
        "uptime": f"{uptime:.2f} seconds",
        "request_count": request_count,
        "prediction_count": prediction_count
    }

@app.get("/metrics",
    summary="System Metrics",
    description="Returns detailed system metrics including performance statistics",
    response_model=MetricsResponse
)
async def metrics():
    """System metrics endpoint"""
    global start_time, request_count, prediction_count, error_count, fraud_count, response_times
    
    uptime = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Get model info
    model_info = {
        "type": type(model).__name__ if model else "None",
        "features": len(pipeline.feature_columns) if pipeline and hasattr(pipeline, 'feature_columns') else 0
    }
    
    return MetricsResponse(
        uptime=uptime,
        request_count=request_count,
        prediction_count=prediction_count,
        error_count=error_count,
        fraud_count=fraud_count,
        avg_response_time=avg_response_time,
        last_5_response_times=response_times[-5:] if response_times else [],
        model_info=model_info
    )

@app.post("/predict", 
    response_model=TransactionResponse,
    summary="Fraud Prediction",
    description="Analyzes a transaction and returns the probability of it being fraudulent",
    response_description="Prediction results including fraud probability"
)
async def predict(request: TransactionRequest):
    """Make fraud predictions"""
    global model, pipeline, prediction_count, error_count, fraud_count
    
    # Generate request ID
    request_id = f"pred-{int(time.time())}-{prediction_count}"
    
    # Record start time
    start_process = time.time()
    
    # Check if model and pipeline are loaded
    if model is None or pipeline is None:
        error_count += 1
        logger.error(f"Request {request_id}: Model or pipeline not loaded")
        raise HTTPException(status_code=503, detail="Model or pipeline not loaded")
    
    try:
        # Log the received data
        transaction_data = request.data
        logger.info(f"Request {request_id}: Received prediction request with {len(transaction_data)} features")
        
        # Convert dict to DataFrame
        data = pd.DataFrame([transaction_data])
        
        # Add hour and day if not present
        current_datetime = datetime.now()
        if 'day' not in data.columns:
            data['day'] = current_datetime.weekday()
        if 'hour' not in data.columns:
            data['hour'] = current_datetime.hour
        
        # Transform the data using pipeline
        try:
            X = pipeline.transform_for_predict(data)
            logger.info(f"Request {request_id}: Data transformed successfully, shape: {X.shape}")
        except Exception as e:
            error_count += 1
            logger.error(f"Request {request_id}: Error transforming data: {e}")
            raise HTTPException(status_code=400, detail=f"Error transforming data: {str(e)}")
        
        # Make prediction
        try:
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0, 1])
            prediction_count += 1
            
            # Increment fraud count if fraud detected
            if prediction == 1:
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
            
            # Log the prediction
            prediction_log = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "transaction": transaction_data,
                "prediction": prediction,
                "probability": probability,
                "is_fraud": bool(prediction == 1),
                "processing_time": processing_time
            }
            prediction_logger.info(json.dumps(prediction_log))
            
            logger.info(f"Request {request_id}: Prediction: {prediction}, Probability: {probability:.4f}, Time: {processing_time:.4f}s")
            
            return result
        except Exception as e:
            error_count += 1
            logger.error(f"Request {request_id}: Error making prediction: {e}")
            raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")
            
    except Exception as e:
        error_count += 1
        logger.error(f"Request {request_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handler for all exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    global error_count
    error_count += 1
    
    # Log the error
    logger.error(f"Global exception: {exc}")
    
    # Return a JSON response
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    port = 8015
    logger.info(f"Starting Enhanced Fraud Detection API on port {port}...")
    uvicorn.run("enhanced_fraud_api:app", host="0.0.0.0", port=port)

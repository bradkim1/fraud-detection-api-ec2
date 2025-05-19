from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Final Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recreate the AdvancedMLPipeline class that was used to save the pipeline
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
#app = FastAPI(title="Final Fraud Detection API")

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
        "message": "Final Fraud Detection API is running.",
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
            raise HTTPException(status_code=503, detail=f"Model or pipeline not loaded: {str(e)}")
    
    try:
        # Log the received data
        logger.info(f"Received prediction request with data keys: {list(request.data.keys())}")
        
        # Convert dict to DataFrame
        data = pd.DataFrame([request.data])
        logger.info(f"Created DataFrame with shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        
        # Add missing columns that might be needed
        current_datetime = pd.Timestamp.now()
        
        # Add day and hour if not present
        if 'day' not in data.columns:
            data['day'] = current_datetime.dayofweek
            logger.info("Added 'day' feature")
        if 'hour' not in data.columns:
            data['hour'] = current_datetime.hour
            logger.info("Added 'hour' feature")
            
        # Feature adjustments logic
        # This is where we'll handle feature engineering and column ordering
            
        # Add derived features
        c_cols = [col for col in data.columns if col.startswith('C')]
        if c_cols and 'C_sum' not in data.columns:
            data['C_sum'] = data[c_cols].sum(axis=1)
            logger.info("Added 'C_sum' feature")
            
        d_cols = [col for col in data.columns if col.startswith('D')]
        if d_cols and 'D_missing' not in data.columns:
            data['D_missing'] = data[d_cols].isnull().sum(axis=1)
            logger.info("Added 'D_missing' feature")
            
        v_cols = [col for col in data.columns if col.startswith('V')]
        if v_cols and 'V_mean' not in data.columns:
            data['V_mean'] = data[v_cols].mean(axis=1)
            logger.info("Added 'V_mean' feature")
            
        # Try direct prediction approach
        try:
            logger.info("Attempting direct prediction with model...")
            
            # Check what features the model expects
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
                logger.info(f"Model expects {len(feature_names)} features")
                
                # Create a DataFrame with all zeros, but with the right columns in the right order
                dummy_data = pd.DataFrame(0, index=[0], columns=feature_names)
                
                # Fill in the dummy data with our real data where columns match
                for col in data.columns:
                    if col in dummy_data.columns:
                        dummy_data[col] = data[col].values
                
                # Use this properly ordered data
                prediction = int(model.predict(dummy_data)[0])
                probability = float(model.predict_proba(dummy_data)[0, 1])
                logger.info(f"Direct prediction made successfully!")
                
                return {
                    "prediction": prediction,
                    "probability": probability,
                    "is_fraud": bool(prediction == 1)
                }
            else:
                logger.info("Model does not have feature_names_in_ attribute. Falling back to pipeline.")
        except Exception as e:
            logger.error(f"Error in direct prediction: {e}")
            logger.info("Falling back to pipeline transformation...")
        
        # Original pipeline approach as fallback
        try:
            X = pipeline.transform_for_predict(data)
            logger.info(f"Transformed data shape: {X.shape}")
            
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0, 1])
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            
            return {
                "prediction": prediction,
                "probability": probability,
                "is_fraud": bool(prediction == 1)
            }
        except Exception as e:
            logger.error(f"Error in pipeline transformation: {e}")
            raise HTTPException(status_code=400, detail=f"Error transforming data: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Final Fraud Detection API on port 8010...")
    uvicorn.run("final_fraud_api:app", host="0.0.0.0", port=8015)

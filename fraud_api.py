from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Recreate the AdvancedMLPipeline class so joblib can load the pickled objects
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

# Load the model and pipeline
MODEL_DIR = '/home/ubuntu/model-ec2/model'
model_path = os.path.join(MODEL_DIR, 'model.pkl')
pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')

try:
    # Since we've recreated the class definition, we can now load the pickle files
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    print("Model and pipeline loaded successfully")
except Exception as e:
    print(f"Error loading model or pipeline: {e}")
    model = None
    pipeline = None

# Create the FastAPI app
app = FastAPI(title="Fraud Detection API")

class TransactionRequest(BaseModel):
    data: Dict[str, Any]

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Model or pipeline not loaded")
    
    try:
        # Convert dict to DataFrame
        data = pd.DataFrame([request.data])
        
        # Transform data for prediction
        X = pipeline.transform_for_predict(data)
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        
        return {
            "prediction": prediction,
            "probability": probability,
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running. Use /predict endpoint."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fraud_api:app", host="0.0.0.0", port=8001)  # Use port 8001 instead of 8000

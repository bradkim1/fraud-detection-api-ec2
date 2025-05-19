import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib
import os
import argparse
import sys

# Set up argument parsing
parser = argparse.ArgumentParser(description='Fraud Detection Model Builder')
parser.add_argument('--sample', action='store_true', help='Use a sample of the data')
parser.add_argument('--model-type', choices=['rf', 'gb', 'xgb'], default='rf', 
                    help='Specify model type: rf, gb, or xgb')
args = parser.parse_args()

# Configure output directory
MODEL_DIR = '/home/ubuntu/model-ec2/model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Advanced ML Pipeline Class
class AdvancedMLPipeline:
    def __init__(self, model_type='rf', n_components=10, remove_outliers=True):
        # Initialize all components: model, imputers, encoder, PCA
        model_map = {
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.model = model_map[model_type]
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

        df['C_sum'] = df[c_cols].sum(axis=1)
        df['D_missing'] = df[d_cols].isnull().sum(axis=1)
        df['V_mean'] = df[v_cols].mean(axis=1)
        return df

    def _remove_outliers(self, df, col='TransactionAmt'):
        # Remove outliers from TransactionAmt using IQR
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = df.shape[0]
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            after = df.shape[0]
            print(f"Removed {before - after} outliers from {col}")
        return df

    def fit_model_on_chunk(self, df, label_column='isFraud'):
        df = self._add_features(df)

        if self.remove_outliers:
            df = self._remove_outliers(df)

        # Convert known categoricals to string
        known_cats = ['ProductCD', 'card4', 'card6', 'DeviceType', 'DeviceInfo',
                      'M1','M2','M3','M4','M5','M6','M7','M8','M9']
        for col in known_cats:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Identify categorical and numeric columns
        self.cat_columns = [col for col in df.columns if df[col].dtype == 'object' and col != label_column]
        num_cols = [col for col in df.columns if col not in self.cat_columns + [label_column, 'TransactionID']]

        # Impute missing values
        if self.cat_columns:
            df[self.cat_columns] = self.imputer_cat.fit_transform(df[self.cat_columns])
        if num_cols:
            df[num_cols] = self.imputer_num.fit_transform(df[num_cols])

        # One-hot encode categorical features
        encoded = self.encoder.fit_transform(df[self.cat_columns])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_columns), index=df.index)

        # Replace original categoricals with encoded version
        df = df.drop(columns=self.cat_columns)
        df = pd.concat([df, encoded_df], axis=1)

        # Scale numeric columns
        if num_cols:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        # Apply PCA on V-columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            pca_trans = self.pca.fit_transform(df[v_cols])
            pca_df = pd.DataFrame(pca_trans, columns=[f'V_PCA_{i}' for i in range(pca_trans.shape[1])], index=df.index)
            df = df.drop(columns=v_cols)
            df = pd.concat([df, pca_df], axis=1)

        # Save the final feature columns
        self.feature_columns = [col for col in df.columns if col not in [label_column, 'TransactionID']]

        # Train model
        X = df[self.feature_columns]
        y = df[label_column]
        self.model.fit(X, y)

    def transform_for_predict(self, df):
        df = self._add_features(df)

        # Fill missing categorical values
        for col in self.cat_columns:
            if col not in df.columns:
                df[col] = "missing"
        df[self.cat_columns] = self.imputer_cat.transform(df[self.cat_columns])

        # Impute numeric
        num_cols = [col for col in df.columns if col not in self.cat_columns + ['TransactionID']]
        df[num_cols] = self.imputer_num.transform(df[num_cols])

        # Encode categoricals
        encoded = self.encoder.transform(df[self.cat_columns])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.cat_columns), index=df.index)

        df = df.drop(columns=self.cat_columns)
        df = pd.concat([df, encoded_df], axis=1)

        # Scale numeric
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        # Apply PCA to V columns
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            pca_trans = self.pca.transform(df[v_cols])
            pca_df = pd.DataFrame(pca_trans, columns=[f'V_PCA_{i}' for i in range(pca_trans.shape[1])], index=df.index)
            df = df.drop(columns=v_cols)
            df = pd.concat([df, pca_df], axis=1)

        # Align columns with training
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        return df

    def evaluate(self, X, y_true):
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        print(classification_report(y_true, y_pred))
        print("ROC AUC:", roc_auc_score(y_true, y_proba))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    def get_metrics(self, X, y_true):
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        return {
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }

# Data Loading Utilities
def get_sample_data(file_path, sample_size=10000, random_state=42):
    """
    Get a random sample from a large dataset
    """
    # Read only first N rows to get column names
    df_temp = pd.read_csv(file_path, nrows=1)
    
    # Get total number of rows
    with open(file_path, 'r') as f:
        num_lines = sum(1 for _ in f) - 1  # Subtract header
    
    # Generate random row indices
    skip_indices = set(np.random.RandomState(random_state).choice(
        range(1, num_lines + 1), 
        size=num_lines - sample_size, 
        replace=False
    ))
    
    # Read data, skipping rows
    df = pd.read_csv(
        file_path,
        skiprows=lambda i: i > 0 and i in skip_indices
    )
    
    print(f"Loaded {df.shape[0]} samples with {df.shape[1]} features.")
    return df

def load_data_in_chunks(file_path, chunk_size=10000):
    """
    Load data in chunks to handle large datasets
    """
    return pd.read_csv(file_path, chunksize=chunk_size)

def create_pipeline_module():
    """
    Create a Python module for easy reuse of the pipeline
    """
    module_path = os.path.join(MODEL_DIR, 'pipeline_module.py')
    
    with open(module_path, 'w') as f:
        f.write("""
import pandas as pd
import joblib

class FraudDetectionPipeline:
    def __init__(self, model_path='/home/ubuntu/model-ec2/model/model.pkl', pipeline_path='/home/ubuntu/model-ec2/model/pipeline.pkl'):
        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
    
    def predict(self, data):
        '''
        Make fraud predictions on new data
        
        Args:
            data: DataFrame or dict with transaction data
            
        Returns:
            tuple: (prediction, probability)
        '''
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        # Prepare data for model
        X = self.pipeline.transform_for_predict(data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return prediction, probability

# Example usage:
# from pipeline_module import FraudDetectionPipeline
# pipeline = FraudDetectionPipeline()
# prediction, probability = pipeline.predict({'TransactionAmt': 100.0, 'card1': 1.0, 'ProductCD': 'C'})
""")
    
    print(f"Created pipeline module at {module_path}")

def create_api_script():
    """
    Create a FastAPI script for serving predictions
    """
    api_path = '/home/ubuntu/model-ec2/app.py'
    
    with open(api_path, 'w') as f:
        f.write("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Union, List
import pandas as pd
from pipeline_module import FraudDetectionPipeline
import uvicorn

app = FastAPI(title="Fraud Detection API")
pipeline = FraudDetectionPipeline()

class TransactionRequest(BaseModel):
    data: Dict[str, Any]

class TransactionResponse(BaseModel):
    prediction: int
    probability: float
    is_fraud: bool

@app.post("/predict", response_model=TransactionResponse)
async def predict(request: TransactionRequest):
    try:
        prediction, probability = pipeline.predict(request.data)
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running. Use /predict endpoint."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
""")
    
    print(f"Created API script at {api_path}")

def main():
    print("\n" + "="*50)
    print("Fraud Detection Model Builder")
    print("="*50)
    
    # Check for data file
    data_file = 'train_transaction.csv'
    if not os.path.exists(data_file):
        print(f"\nERROR: Data file '{data_file}' not found!")
        print(f"Please upload your fraud detection dataset to the current directory.")
        print(f"Expected file: {data_file}")
        sys.exit(1)
    
    # Initialize pipeline with specified model type
    print(f"\nInitializing model pipeline with {args.model_type.upper()} model type...")
    pipeline = AdvancedMLPipeline(model_type=args.model_type, n_components=10, remove_outliers=True)
    
    # Train on sample or full dataset
    if args.sample:
        print("\nTraining on a SAMPLE of the dataset...")
        # Get a sample of the data
        train_df = get_sample_data(data_file, sample_size=20000)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            train_df.drop('isFraud', axis=1),
            train_df['isFraud'],
            test_size=0.2,
            random_state=42,
            stratify=train_df['isFraud']
        )
        
        # Train the model
        print("\nFitting model...")
        pipeline.fit_model_on_chunk(X_train.assign(isFraud=y_train), label_column='isFraud')
        
        # Evaluate
        print("\nEvaluating model...")
        X_val_transformed = pipeline.transform_for_predict(X_val)
        pipeline.evaluate(X_val_transformed, y_val)
        
        # Print metrics
        metrics = pipeline.get_metrics(X_val_transformed, y_val)
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
    else:
        print("\nTraining on the FULL dataset (this may take a while)...")
        chunks = load_data_in_chunks(data_file, chunk_size=50000)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}...")
            if i == 0:
                # First chunk: fit the model
                pipeline.fit_model_on_chunk(chunk, label_column='isFraud')
            else:
                # Later chunks: incrementally fit if your model supports it
                X_trans = pipeline.transform_for_predict(chunk.drop('isFraud', axis=1))
                y_chunk = chunk['isFraud']
                pipeline.model.fit(X_trans, y_chunk)
        
        # Set aside a validation set from the last chunk
        X_val, y_val = chunk.drop('isFraud', axis=1), chunk['isFraud']
        X_val_transformed = pipeline.transform_for_predict(X_val)
        
        # Evaluate
        print("\nEvaluating model...")
        pipeline.evaluate(X_val_transformed, y_val)
    
    # Save model and pipeline
    print("\nSaving model and pipeline...")
    joblib.dump(pipeline.model, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'pipeline.pkl'))
    
    # Create additional files
    create_pipeline_module()
    create_api_script()
    
    print("\nDone! Your fraud detection model has been successfully built and saved.")
    print(f"- Model saved to: {os.path.join(MODEL_DIR, 'model.pkl')}")
    print(f"- Pipeline saved to: {os.path.join(MODEL_DIR, 'pipeline.pkl')}")
    print("\nTo use the API for predictions:")
    print("1. Run: python /home/ubuntu/model-ec2/app.py")
    print("2. Access the API at: http://your-ec2-ip:8000/docs")

if __name__ == "__main__":
    main()

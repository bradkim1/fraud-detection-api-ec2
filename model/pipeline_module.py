
import pandas as pd
import joblib

class FraudDetectionPipeline:
    def __init__(self, model_path='/home/ubuntu/model-ec2/model/model.pkl', pipeline_path='/home/ubuntu/model-ec2/model/pipeline.pkl'):
        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
    
    def predict(self, data):
        """
        Make fraud predictions on new data
        
        Args:
            data: DataFrame or dict with transaction data
            
        Returns:
            tuple: (prediction, probability)
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        # Prepare data for model
        X = self.pipeline.transform_for_predict(data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return prediction, probability

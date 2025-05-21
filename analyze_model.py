import joblib
import pandas as pd
import numpy as np
import json

# Load the model and pipeline
model_path = "./model/model.pkl"
pipeline_path = "./model/pipeline.pkl"

try:
    print("Loading model...")
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Model requires {len(model.feature_names_in_)} features:")
        # Print first 10 and last 10 features
        print("First 10 features:")
        print(model.feature_names_in_[:10])
        print("Last 10 features:")
        print(model.feature_names_in_[-10:])
        
        # Save all features to a file
        with open("model_features.json", "w") as f:
            json.dump(model.feature_names_in_.tolist(), f, indent=2)
        print("Saved all feature names to model_features.json")
    
    print("\nLoading pipeline...")
    pipeline = joblib.load(pipeline_path)
    print(f"Pipeline type: {type(pipeline)}")
    
    # Create a sample input with all zeros
    if hasattr(model, 'feature_names_in_'):
        # Create a DataFrame with all required features
        feature_names = model.feature_names_in_
        sample_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        
        print("\nTesting prediction with all-zero sample...")
        try:
            # Try transforming with pipeline
            if hasattr(pipeline, 'transform_for_predict'):
                X = pipeline.transform_for_predict(sample_data)
                print("Pipeline transformation successful")
            else:
                X = pipeline.transform(sample_data)
                print("Pipeline transformation successful")
                
            # Try prediction with model
            prediction = model.predict(X)
            print(f"Model prediction successful: {prediction[0]}")
            
            proba = model.predict_proba(X)
            print(f"Fraud probability: {proba[0][1]:.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")
    
except Exception as e:
    print(f"Error: {e}")

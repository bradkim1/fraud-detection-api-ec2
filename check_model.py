import joblib
import os

# Define paths
MODEL_DIR = '/home/ubuntu/model-ec2/model'
model_path = os.path.join(MODEL_DIR, 'model.pkl')
pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')

# Check model
print("Checking model...")
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully! Type: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")

# Check pipeline
print("\nChecking pipeline...")
try:
    pipeline = joblib.load(pipeline_path)
    print(f"Pipeline loaded successfully! Type: {type(pipeline)}")
    
    # Check pipeline methods
    print("\nPipeline attributes and methods:")
    print([attr for attr in dir(pipeline) if not attr.startswith('_')])
    
    # Check if transform_for_predict exists
    if hasattr(pipeline, 'transform_for_predict'):
        print("\ntransform_for_predict method exists!")
    else:
        print("\ntransform_for_predict method NOT found")
except Exception as e:
    print(f"Error loading pipeline: {e}")

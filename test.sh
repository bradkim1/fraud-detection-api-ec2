# Navigate to your directory
cd /home/ubuntu/model-ec2

# Create a model training script
cat > train_model.py << 'EOL'
#!/usr/bin/env python3
"""
Fraud Detection Model Training Script

This script trains the fraud detection model based on the AdvancedMLPipeline class.
"""

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
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("Warning: XGBoost not installed. XGB model type will not be available.")
    HAS_XGB = False
import joblib
import os
import sys

# Define the AdvancedMLPipeline class
class AdvancedMLPipeline:
    def __init__(self, model_type='rf', n_components=10, remove_outliers=True):
        #

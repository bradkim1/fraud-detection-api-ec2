#!/usr/bin/env python3

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
        # Initialize all components: model, imputers, encoder, PCA
        model_map = {
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        if HAS_XGB:
            model_map['xgb'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            
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

        if c_cols:
            df['C_sum'] = df[c_cols].sum(axis=1)
        if d_cols:
            df['D_missing'] = df[d_cols].isnull().sum(axis=1)
        if v_cols:
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

        # Save the f

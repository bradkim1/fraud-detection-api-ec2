#!/bin/bash
# Script to download required large files for the fraud detection system

# Create directories if they don't exist
mkdir -p model
mkdir -p notebooks

echo "This script will download large files needed for the fraud detection system"
echo "--------------------------------"

# Download Miniconda if needed
if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
  echo "Downloading Miniconda installer..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  echo "Miniconda installer downloaded"
fi

# Download training dataset (replace with actual URL)
if [ ! -f "notebooks/train_transaction.csv" ]; then
  echo "Downloading transaction dataset..."
  # Replace with actual dataset URL
  echo "Please download the transaction dataset manually from [SOURCE URL]"
  echo "and place it in the notebooks/ directory"
  # Or use wget if you have a direct download link:
  # wget -O notebooks/train_transaction.csv https://example.com/datasets/train_transaction.csv
fi

echo "--------------------------------"
echo "Downloads complete. Please train the model using notebooks/fraud_model_training.ipynb"

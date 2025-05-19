# Fraud Detection API

A machine learning API for detecting fraudulent transactions. This system uses a Random Forest classifier trained on transactional data to identify potentially fraudulent activities.

![Fraud Detection UI](docs/images/fraud-detection-ui.png)

## Features

- **Advanced ML Model**: Trained on transaction data to detect fraudulent patterns
- **Real-time API**: Fast response times for immediate fraud assessment
- **Interactive UI**: User-friendly interface for submitting transactions and visualizing results
- **Monitoring Dashboard**: Track system performance and fraud metrics
- **Comprehensive Logging**: Detailed logs for all system events

## Installation

### Prerequisites

- Python 3.8+
- pip
- Docker (optional)

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bradkim1/fraud-detection-api-ec2.git
   cd fraud-detection-api-ec2

## Note about model files

The model files (*.pkl) are not included in this repository due to their size. Please see the instructions in the documentation for generating or downloading these files.

## Note about model files

The model files (*.pkl) are not included in this repository due to their size. Please see the instructions in the documentation for generating or downloading these files.

## Large Files Not Included in Repository

This repository does not include the following large files due to GitHub size limitations:

1. **Model Files (`*.pkl`)**:
   - Users need to train their own models using the provided notebook
   - See `notebooks/fraud_model_training.ipynb` for the training process

2. **Training Dataset (`train_transaction.csv`)**:
   - This large dataset needs to be downloaded separately
   - Place it in the `notebooks/` directory before running the training notebook

3. **Miniconda Installer**:
   - Not required for running the application
   - If needed, download from https://docs.conda.io/en/latest/miniconda.html

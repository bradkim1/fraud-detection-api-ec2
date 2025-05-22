# Fraud Detection API

A machine learning API for detecting fraudulent transactions. This system uses a Random Forest classifier trained on transactional data to identify potentially fraudulent activities.

![Fraud Detection UI](docs/source/images/prediction_interface.png)

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


**Training Dataset train_transaction.csv
   - This needs to be unzipped when needed for model training

## Large File Not Included in Repository

This repository does not include the following large file due to GitHub size limitations:


1. **Miniconda Installer**:
   - Not required for running the application
   - If needed, download from https://docs.conda.io/en/latest/miniconda.html
  
     
Fraud Detection API project follows this structure:

All other files which were committed but not on the structure were used for test purpose

    model-ec2/
    ├── model/                    # Trained model and pipeline files
    │   ├── model.pkl             # Serialized machine learning model
    │   ├── pipeline.pkl          # Preprocessing pipeline
    │   └── pipeline_module.py    # Helper module for using the pipeline
    ├── fraud-ui/                 # Web interface files
    │   ├── index.html            # Main prediction interface
    │   ├── dashboard.html        # Monitoring dashboard
    │   └── logs.html             # Log viewer
    ├── docs/                     # Documentation
    │   ├── source/               # Documentation source files
    │   └── build/                # Generated documentation
    ├── notebooks/                # Jupyter notebooks
    │   └── fraud_model_training.ipynb # Model training notebook
    ├── logs/                     # API logs
    ├── nginx/                    # Nginx configuration
    ├── enhanced_fraud_api.py     # API implementation
    ├── requirements.txt          # Python dependencies
    ├── Dockerfile                # Docker configuration
    ├── docker-compose.yml        # Docker Compose configuration
    └── README.md                 # Project documentation


User Guide: Fraud Detection Dashboard

This is comprehensive guide for using the UI dashboard:

Getting Started


Accessing the Dashboard

1. Open web browser
2. Navigate to deployed application URL: 
	- EC2 deployment: http://3.142.239.148

Dashboard Overview


Main Interface Components

1. Header Section
	- Application title: "Fraud Detection System"
	- Navigation menu 
	- Current status indicator
2. Input Panel
	- Transaction input form
	  
3. Results Panel
	- Prediction results
	- Confidence scores
	- Risk assessment
4. Analytics Section 
	- Recent predictions summary
	- Performance metrics
	- Visual charts

How to Use the Dashboard


Single Transaction Analysis

1. Enter Transaction Details:

Example:
.Transaction Amount: $150.00
.Card ID: 1111222233334444
.Card category: Credit (or Debit)
.Product Category: W (or select from dropdown)
.Card Type: Visa (or select from dropdown)
.Transaction Date/Time: 05/22/2025, 05:36 AM
.Purchaser email domain: gmail.com (select from dropdown)
  
2. Submit for Analysis:
	- Click the "Analyze Transaction" button
	- Wait for processing (usually 1-2 seconds)
3. Interpret Results:
	- Green: Low fraud risk (< 30% probability)
	- Yellow: Medium fraud risk (30-70% probability)
	- Red: High fraud risk (> 70% probability)

Understanding the Results


Fraud Probability Score

- 0.0 - 0.3: Low Risk (Safe to approve)
- 0.3 - 0.7: Medium Risk (Review required)
- 0.7 - 1.0: High Risk (Likely fraud - decline/investigate)

Common Use Cases


1. Real-time Transaction Monitoring

```
Use Case: Processing live transactions
Steps:
1. Enter transaction details as they occur
2. Get instant fraud assessment
3. Make approval/decline decision based on risk score
```

2. Use Case: Analyzing past transactions for patterns

```
Steps:
1. Upload historical transaction file
2. Review batch results
3. Identify trends and patterns
4. Ad
```

Features and Functionality


Input Validation

- The system validates input data before processing
- Error messages appear for invalid inputs: 
	- Negative transaction amounts
	- Invalid date formats
	- Missing required fields

Response Time

- Typical response time: 1-3 seconds
  

Export Options

- Individual Results: Copy/paste or screenshot
- Analytics: Export charts and summaries

Troubleshooting


Common Issues and Solutions

1. "Model Not Found" Error
	- Cause: Backend API is not running or model file is missing
	- Solution: Contact system administrator
2. Slow Response Times
	- Cause: High server load or large batch processing
	- Solution: Try smaller batches or wait for current processing to complete
3. Invalid Input Format
	- Cause: Data doesn't match expected format
	- Solution: Check input requirements and format data correctly
4. Connection Errors
	- Cause: Network issues or server downtime
	- Solution: Refresh page and try again, or contact support



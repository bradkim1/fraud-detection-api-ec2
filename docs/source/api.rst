API Documentation
================

This section provides detailed documentation for the Fraud Detection API endpoints, models, and data structures.

Fraud Detection API
------------------

The API provides several endpoints for fraud detection and system monitoring:

Health Check Endpoint
^^^^^^^^^^^^^^^^^^^^

**GET /**

Returns the current status of the API, including:

* Message confirming the API is running
* Model load status
* Pipeline load status
* Uptime in seconds
* Request count
* Prediction count

Example response:

.. code-block:: json

    {
        "message": "Enhanced Fraud Detection API is running.",
        "model_status": "loaded",
        "pipeline_status": "loaded",
        "uptime": "3600.15 seconds",
        "request_count": 50,
        "prediction_count": 25
    }

Metrics Endpoint
^^^^^^^^^^^^^^^

**GET /metrics**

Returns detailed system metrics including:

* API uptime
* Request count
* Prediction count
* Error count
* Fraud count
* Average response time
* Recent response times
* Model information

Example response:

.. code-block:: json

    {
        "uptime": 3600.15,
        "request_count": 50,
        "prediction_count": 25,
        "error_count": 2,
        "fraud_count": 5,
        "avg_response_time": 0.156,
        "last_5_response_times": [0.145, 0.162, 0.158, 0.149, 0.167],
        "model_info": {
            "type": "RandomForestClassifier",
            "features": 193
        }
    }

Prediction Endpoint
^^^^^^^^^^^^^^^^^

**POST /predict**

Analyzes a transaction and returns the probability of it being fraudulent.

Request format:

.. code-block:: json

    {
        "data": {
            "TransactionAmt": 100.0,
            "ProductCD": "C",
            "card1": 1234,
            "card4": "visa",
            "card6": "debit",
            "P_emaildomain": "gmail.com"
        }
    }

Response format:

.. code-block:: json

    {
        "prediction": 0,
        "probability": 0.12,
        "is_fraud": false,
        "request_id": "pred-1621415412-0",
        "processing_time": 0.156
    }

Documentation Endpoints
^^^^^^^^^^^^^^^^^^^^^^

**GET /docs**

Access the Swagger UI documentation.

**GET /openapi.json**

Access the OpenAPI specification in JSON format.

Models
------

Fraud Detection Model
^^^^^^^^^^^^^^^^^^^^

The API uses a Random Forest classifier trained on transaction data. The model:

* Was trained on a dataset containing both legitimate and fraudulent transactions
* Uses 100 decision trees with bootstrap sampling
* Evaluates multiple features to identify patterns indicative of fraud
* Outputs both a binary classification (0=legitimate, 1=fraud) and a fraud probability score

The model achieves:

* F1 score: 0.92
* Precision: 0.95
* Recall: 0.89
* ROC AUC: 0.97

Feature Engineering Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The feature engineering pipeline performs several transformations:

1. **Feature Creation**:
   * Converts TransactionDT to hour and day features
   * Creates aggregated features (C_sum, D_missing, V_mean)

2. **Missing Value Imputation**:
   * Numerical features: Mean imputation
   * Categorical features: Most frequent value imputation

3. **Categorical Encoding**:
   * One-hot encoding for categorical variables
   * Special handling for unknown categories

4. **Numerical Scaling**:
   * StandardScaler to normalize numerical features

5. **Dimensionality Reduction**:
   * PCA on V-columns to reduce dimensionality

6. **Outlier Handling**:
   * IQR-based outlier removal for TransactionAmt

Data Models
----------

TransactionRequest
^^^^^^^^^^^^^^^^^

The request format for submitting transactions to the API:

.. code-block:: python

    class TransactionRequest:
        """Transaction data for fraud prediction"""
        data: Dict[str, Any]  # Transaction details including amount, card info, etc.

Required fields in the data dictionary:

* **TransactionAmt**: float - The transaction amount
* **ProductCD**: string - Product code (e.g., "C", "H", "R", "S", "W")
* **card1**: int - Card identifier

Optional fields (improve prediction accuracy):

* **card4**: string - Card type (e.g., "visa", "mastercard")
* **card6**: string - Card category (e.g., "debit", "credit")
* **P_emaildomain**: string - Purchaser email domain
* Additional C, D, M, V features if available

TransactionResponse
^^^^^^^^^^^^^^^^^

The response format for predictions:

.. code-block:: python

    class TransactionResponse:
        """Fraud prediction response"""
        prediction: int       # Binary classification (0=legitimate, 1=fraud)
        probability: float    # Fraud probability between 0-1
        is_fraud: bool        # Boolean flag indicating fraud detection
        request_id: str       # Unique identifier for the request
        processing_time: float  # Time taken to process the request in seconds

MetricsResponse
^^^^^^^^^^^^^

The response format for the metrics endpoint:

.. code-block:: python

    class MetricsResponse:
        """System metrics response"""
        uptime: float         # API uptime in seconds
        request_count: int    # Total number of requests
        prediction_count: int # Total number of predictions
        error_count: int      # Total number of errors
        fraud_count: int      # Total number of fraud detections
        avg_response_time: float  # Average response time in seconds
        last_5_response_times: List[float]  # Last 5 response times
        model_info: Dict[str, Any]  # Model information

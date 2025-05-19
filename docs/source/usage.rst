Usage
=====

API Endpoints
------------

The Fraud Detection API provides the following endpoints:

- **GET /** - Health check endpoint
- **GET /metrics** - System metrics endpoint
- **POST /predict** - Make fraud predictions
- **GET /docs** - API documentation (Swagger UI)

Making Predictions
-----------------

To make a prediction, send a POST request to the `/predict` endpoint with a JSON payload.

Example request:

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

Example response:

.. code-block:: json

   {
     "prediction": 0,
     "probability": 0.12,
     "is_fraud": false,
     "request_id": "pred-1621415412-0",
     "processing_time": 0.156
   }

Python client example:

.. code-block:: python

   import requests
   
   API_URL = "http://localhost:8015/predict"
   
   data = {
       "data": {
           "TransactionAmt": 100.0,
           "ProductCD": "C",
           "card1": 1234,
           "card4": "visa",
           "card6": "debit",
           "P_emaildomain": "gmail.com"
       }
   }
   
   response = requests.post(API_URL, json=data)
   result = response.json()
   
   print(f"Prediction: {'Fraud' if result['is_fraud'] else 'Legitimate'}")
   print(f"Probability: {result['probability'] * 100:.1f}%")

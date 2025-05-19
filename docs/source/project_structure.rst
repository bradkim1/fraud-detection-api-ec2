Project Structure
================

Our Fraud Detection API project follows this structure:

.. code-block:: text

    fraud-detection-api/
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

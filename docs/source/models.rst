Model Documentation
==================

Fraud Detection Model
-------------------

The fraud detection model is a Random Forest classifier trained on transaction data. It analyzes various features of a transaction to determine the likelihood of fraud.

Model Features
------------

The model uses the following types of features:

* Transaction amount
* Card information
* Product codes
* Email domains
* Time-based features
* Device information

Feature Engineering
-----------------

The `AdvancedMLPipeline` class handles feature engineering:

* Missing value imputation
* Categorical encoding
* Numerical scaling
* Dimensionality reduction
* Feature creation

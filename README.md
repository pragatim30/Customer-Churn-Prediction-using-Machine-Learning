# Customer Churn Prediction using Machine Learning

## Overview

Customer churn refers to a customer discontinuing their services. Predicting churn can help businesses identify customers who may leave and understand the factors associated with customer retention.

This project develops a machine learning classification system to predict whether a customer is likely to churn.

## Objective

The objective is to build and compare multiple classical machine learning models for customer churn prediction.

## Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Churn Prediction
```

## Models Used

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Key Concepts Demonstrated

* Data preprocessing
* Missing-value handling
* Exploratory Data Analysis
* Categorical encoding
* Feature scaling
* Train-test splitting
* Supervised learning
* Classification
* Model evaluation
* Confusion matrix
* Feature importance
* Model comparison

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── customer_churn.csv
│
├── Customer_Churn_Prediction.ipynb
│
├── requirements.txt
│
└── README.md
```

## Results

The four machine learning models are compared using multiple evaluation metrics. The final results are generated directly from the test dataset and reported in the notebook.

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Handling class imbalance
* Feature selection
* Deployment using Streamlit
* Real-time customer churn prediction API
* Model monitoring

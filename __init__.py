"""
Titanic Survival Prediction - Source Package

This package contains modules for data preprocessing, feature engineering,
model training, and prediction for the Titanic survival prediction project.
"""

from .data_preprocessing import preprocess_data, load_data
from .feature_engineering import engineer_features
from .model_training import get_models, train_and_evaluate
from .predict import predict_single_passenger, batch_predict

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'preprocess_data',
    'load_data',
    'engineer_features',
    'get_models',
    'train_and_evaluate',
    'predict_single_passenger',
    'batch_predict'
]

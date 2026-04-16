"""
Data Preprocessing Module for Titanic Survival Prediction

This module contains functions for cleaning and preprocessing
the Titanic dataset before model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(filepath):
    """
    Load the Titanic dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    return pd.read_csv(filepath)


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with missing values
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df = df.copy()
    
    # Fill Age with median grouped by Pclass and Sex
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Fill Fare with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    return df


def create_cabin_features(df):
    """
    Create features from Cabin column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with new cabin features
    """
    df = df.copy()
    
    # Create Has_Cabin feature
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # Extract Deck from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
    
    return df


def encode_categorical(df):
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with categorical variables
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical variables
    """
    df = df.copy()
    
    # Label encode Sex
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    # One-hot encode Embarked and Deck
    df = pd.get_dummies(df, columns=['Embarked', 'Deck'], 
                        prefix=['Embarked', 'Deck'])
    
    return df


def preprocess_data(filepath):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to the raw data CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe ready for feature engineering
    """
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create cabin features
    df = create_cabin_features(df)
    
    # Encode categorical variables
    df = encode_categorical(df)
    
    print(f"✅ Preprocessing completed. Shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test the preprocessing pipeline
    df_processed = preprocess_data('../data/Titanic-Dataset.csv')
    print("\nFirst 5 rows:")
    print(df_processed.head())
    print("\nMissing values after preprocessing:")
    print(df_processed.isnull().sum().sum())

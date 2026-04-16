"""
Feature Engineering Module for Titanic Survival Prediction

This module contains functions for creating new features
from the preprocessed Titanic dataset.
"""

import pandas as pd
import numpy as np


def create_family_features(df):
    """
    Create family-related features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with family features
    """
    df = df.copy()
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is Alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    return df


def create_age_groups(df):
    """
    Create age group feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with age group feature
    """
    df = df.copy()
    
    df['AgeGroup'] = pd.cut(df['Age'], 
                            bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    return df


def create_fare_groups(df):
    """
    Create fare group feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with fare group feature
    """
    df = df.copy()
    
    df['FareGroup'] = pd.qcut(df['Fare'], q=4, 
                               labels=['Low', 'Medium', 'High', 'Premium'])
    
    return df


def extract_title(df):
    """
    Extract title from passenger name.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with title feature
    """
    df = df.copy()
    
    # Extract title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna('Rare')
    
    return df


def encode_new_features(df):
    """
    One-hot encode the new categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded features
    """
    df = df.copy()
    
    # One-hot encode AgeGroup, FareGroup, and Title
    df = pd.get_dummies(df, columns=['AgeGroup', 'FareGroup', 'Title'], 
                        prefix=['Age', 'Fare', 'Title'])
    
    return df


def select_features(df, drop_columns=None):
    """
    Select features for modeling by dropping unnecessary columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    drop_columns : list
        List of columns to drop (default: ['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with selected features
    """
    df = df.copy()
    
    if drop_columns is None:
        drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    
    df = df.drop(columns=drop_columns, errors='ignore')
    
    return df


def engineer_features(df, for_prediction=False):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    for_prediction : bool
        If True, don't expect 'Survived' column
        
    Returns:
    --------
    tuple
        (X, y) if not for_prediction, else X
    """
    # Create family features
    df = create_family_features(df)
    
    # Create age groups
    df = create_age_groups(df)
    
    # Create fare groups
    df = create_fare_groups(df)
    
    # Extract title
    df = extract_title(df)
    
    # Encode new features
    df = encode_new_features(df)
    
    # Select features for modeling
    df = select_features(df)
    
    if for_prediction:
        return df
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    print(f"✅ Feature engineering completed. Features: {X.shape[1]}")
    
    return X, y


if __name__ == "__main__":
    # Test feature engineering
    from data_preprocessing import preprocess_data
    
    df_processed = preprocess_data('../data/Titanic-Dataset.csv')
    X, y = engineer_features(df_processed)
    
    print(f"\nFeature columns ({len(X.columns)}):")
    print(X.columns.tolist())
    print(f"\nTarget distribution:\n{y.value_counts()}")

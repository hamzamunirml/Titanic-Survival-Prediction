"""
Prediction Module for Titanic Survival Prediction

This script loads a trained model and makes predictions on new data.
"""

import pickle
import pandas as pd
import numpy as np
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features


def load_model_and_scaler(model_path='../models/random_forest_model.pkl',
                          scaler_path='../models/scaler.pkl',
                          feature_columns_path='../models/feature_columns.pkl'):
    """
    Load the trained model, scaler, and feature columns.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    scaler_path : str
        Path to the saved scaler
    feature_columns_path : str
        Path to the saved feature columns
        
    Returns:
    --------
    tuple
        (model, scaler, feature_columns)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(feature_columns_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, feature_columns


def predict_single_passenger(passenger_data, model, scaler, feature_columns):
    """
    Predict survival for a single passenger.
    
    Parameters:
    -----------
    passenger_data : dict
        Dictionary containing passenger information
    model : trained model
    scaler : fitted StandardScaler
    feature_columns : list
        List of feature column names
        
    Returns:
    --------
    tuple
        (prediction, probability)
    """
    # Convert to DataFrame
    df_pred = pd.DataFrame([passenger_data])
    
    # Preprocess
    df_pred = preprocess_data_for_prediction(df_pred)
    
    # Engineer features
    df_pred = engineer_features(df_pred, for_prediction=True)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df_pred.columns:
            df_pred[col] = 0
    
    # Reorder columns
    df_pred = df_pred[feature_columns]
    
    # Scale features
    df_pred_scaled = scaler.transform(df_pred)
    
    # Predict
    prediction = model.predict(df_pred_scaled)[0]
    probability = model.predict_proba(df_pred_scaled)[0][1]
    
    return prediction, probability


def preprocess_data_for_prediction(df):
    """
    Preprocess data for prediction (simplified version).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Handle missing values
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna('S', inplace=True)
    
    # Create cabin features
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
    
    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    
    # One-hot encode Embarked and Deck
    df = pd.get_dummies(df, columns=['Embarked', 'Deck'], 
                        prefix=['Embarked', 'Deck'])
    
    return df


def create_sample_passenger():
    """
    Create a sample passenger for demonstration.
    
    Returns:
    --------
    dict
        Sample passenger data
    """
    return {
        'PassengerId': 892,
        'Pclass': 1,
        'Name': 'Doe, Mr. John',
        'Sex': 'male',
        'Age': 30,
        'SibSp': 0,
        'Parch': 0,
        'Ticket': 'PC 12345',
        'Fare': 50.0,
        'Cabin': 'C85',
        'Embarked': 'C'
    }


def batch_predict(filepath, model, scaler, feature_columns):
    """
    Make predictions on a batch of passengers.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with passenger data
    model : trained model
    scaler : fitted StandardScaler
    feature_columns : list
        List of feature column names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(filepath)
    passenger_ids = df['PassengerId'].copy()
    
    # Preprocess
    df = preprocess_data_for_prediction(df)
    
    # Engineer features
    df = engineer_features(df, for_prediction=True)
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions,
        'Survival_Probability': probabilities
    })
    
    return results


def main():
    """
    Main prediction function.
    """
    print("🚀 Titanic Survival Prediction")
    print("=" * 60)
    
    # Load model and scaler
    print("\nLoading model and scaler...")
    try:
        model, scaler, feature_columns = load_model_and_scaler()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model files not found. Please run model_training.py first.")
        return
    
    # Example 1: Single passenger prediction
    print("\n" + "=" * 60)
    print("Example 1: Single Passenger Prediction")
    print("=" * 60)
    
    sample_passenger = create_sample_passenger()
    print("\nPassenger Details:")
    for key, value in sample_passenger.items():
        print(f"  {key}: {value}")
    
    prediction, probability = predict_single_passenger(
        sample_passenger, model, scaler, feature_columns
    )
    
    print(f"\n🔮 Prediction:")
    print(f"   Survived: {'Yes' if prediction == 1 else 'No'}")
    print(f"   Survival Probability: {probability:.2%}")
    
    # Example 2: Another passenger (female, 1st class)
    print("\n" + "=" * 60)
    print("Example 2: Female, 1st Class Passenger")
    print("=" * 60)
    
    female_passenger = {
        'PassengerId': 893,
        'Pclass': 1,
        'Name': 'Smith, Mrs. Jane',
        'Sex': 'female',
        'Age': 28,
        'SibSp': 1,
        'Parch': 0,
        'Ticket': 'PC 54321',
        'Fare': 80.0,
        'Cabin': 'B28',
        'Embarked': 'S'
    }
    
    print("\nPassenger Details:")
    for key, value in female_passenger.items():
        print(f"  {key}: {value}")
    
    prediction, probability = predict_single_passenger(
        female_passenger, model, scaler, feature_columns
    )
    
    print(f"\n🔮 Prediction:")
    print(f"   Survived: {'Yes' if prediction == 1 else 'No'}")
    print(f"   Survival Probability: {probability:.2%}")
    
    # Example 3: Child passenger
    print("\n" + "=" * 60)
    print("Example 3: Child Passenger (3rd class)")
    print("=" * 60)
    
    child_passenger = {
        'PassengerId': 894,
        'Pclass': 3,
        'Name': 'Johnson, Master. Tom',
        'Sex': 'male',
        'Age': 5,
        'SibSp': 1,
        'Parch': 1,
        'Ticket': 'A/5 1234',
        'Fare': 15.0,
        'Cabin': np.nan,
        'Embarked': 'S'
    }
    
    print("\nPassenger Details:")
    for key, value in child_passenger.items():
        print(f"  {key}: {value}")
    
    prediction, probability = predict_single_passenger(
        child_passenger, model, scaler, feature_columns
    )
    
    print(f"\n🔮 Prediction:")
    print(f"   Survived: {'Yes' if prediction == 1 else 'No'}")
    print(f"   Survival Probability: {probability:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Prediction examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

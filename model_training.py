"""
Model Training Module for Titanic Survival Prediction

This script trains multiple machine learning models and saves the best one.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Import custom modules
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features


def get_models():
    """
    Get dictionary of models to train.
    
    Returns:
    --------
    dict
        Dictionary of model names and instances
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and instances
    X_train, X_test : array-like
        Training and testing features
    y_train, y_test : array-like
        Training and testing targets
        
    Returns:
    --------
    dict
        Dictionary containing results for each model
    """
    results = {}
    
    print("=" * 60)
    print("Training Models...")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results


def print_results_table(results):
    """
    Print results in a formatted table.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    """
    # Create results dataframe
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()],
        'CV Mean': [results[m]['cv_mean'] for m in results.keys()],
        'CV Std': [results[m]['cv_std'] for m in results.keys()]
    })
    
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "=" * 60)
    print("Model Performance Summary")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    return results_df


def save_best_model(results, results_df, scaler, feature_columns):
    """
    Save the best performing model.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    results_df : pd.DataFrame
        DataFrame with sorted results
    scaler : StandardScaler
        Fitted scaler
    feature_columns : list
        List of feature column names
    """
    best_model_name = results_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   ROC-AUC:  {results[best_model_name]['roc_auc']:.4f}")
    
    # Save model
    model_filename = f'../models/{best_model_name.replace(" ", "_").lower()}_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler
    scaler_filename = '../models/scaler.pkl'
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    feature_columns_filename = '../models/feature_columns.pkl'
    with open(feature_columns_filename, 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print(f"\n✅ Model saved: {model_filename}")
    print(f"✅ Scaler saved: {scaler_filename}")
    print(f"✅ Feature columns saved: {feature_columns_filename}")
    
    return best_model_name


def main():
    """
    Main training pipeline.
    """
    print("🚀 Starting Model Training Pipeline\n")
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    df_processed = preprocess_data('../data/Titanic-Dataset.csv')
    
    # Step 2: Feature engineering
    print("\nStep 2: Engineering features...")
    X, y = engineer_features(df_processed)
    feature_columns = X.columns.tolist()
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")
    
    # Step 4: Scale features
    print("\nStep 4: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 5: Get models
    print("\nStep 5: Loading models...")
    models = get_models()
    print(f"   {len(models)} models loaded")
    
    # Step 6: Train and evaluate
    print("\nStep 6: Training and evaluating models...")
    results = train_and_evaluate(models, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 7: Print results
    results_df = print_results_table(results)
    
    # Step 8: Save best model
    print("\nStep 7: Saving best model...")
    best_model_name = save_best_model(results, results_df, scaler, feature_columns)
    
    # Step 9: Detailed analysis of best model
    print("\n" + "=" * 60)
    print(f"Detailed Analysis: {best_model_name}")
    print("=" * 60)
    
    best_model = results[best_model_name]['model']
    y_pred = results[best_model_name]['predictions']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Did Not Survive', 'Survived']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\n✅ Training pipeline completed!")


if __name__ == "__main__":
    main()

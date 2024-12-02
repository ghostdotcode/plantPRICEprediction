import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model(X, y, n_estimators=100, test_size=0.2):
    """Train the Random Forest model and return metrics and feature importance."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Calculate feature importance
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    return metrics, feature_importance

def predict_price(input_data):
    """Make prediction for new input data."""
    # Create and train model with default parameters
    # In production, you'd want to load a pre-trained model
    from utils import load_sample_data, preprocess_data
    
    df = load_sample_data()
    X, y = preprocess_data(df)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    
    return model.predict(input_data)[0]

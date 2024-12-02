import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_sample_data():
    """Generate sample data for the application."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'air_humidity': np.random.uniform(30, 90, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'soil_humidity': np.random.uniform(20, 80, n_samples),
        'soil_ph': np.random.uniform(5.5, 8.5, n_samples),
        'rainfall': np.random.uniform(50, 300, n_samples),
        'price': np.zeros(n_samples)
    }
    
    # Generate prices based on realistic relationships
    for i in range(n_samples):
        base_price = 100
        # Higher temperatures generally decrease price
        temp_factor = -0.5 * (data['temperature'][i] - 25)**2
        # Optimal soil pH is around 7
        ph_factor = -20 * (data['soil_ph'][i] - 7)**2
        # Moderate rainfall is good
        rain_factor = -0.001 * (data['rainfall'][i] - 150)**2
        # Moderate humidity is preferred
        humidity_factor = -0.01 * (data['air_humidity'][i] - 60)**2
        
        price = base_price + temp_factor + ph_factor + rain_factor + humidity_factor
        data['price'][i] = max(price, 20)  # Ensure minimum price
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for model training."""
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    
    return X_scaled, y

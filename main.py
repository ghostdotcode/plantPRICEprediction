import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import train_model, predict_price
from utils import load_sample_data, preprocess_data

st.set_page_config(page_title="Crop Price Prediction", layout="wide")

# App title and description
st.title("🌾 Crop Price Prediction Model")
st.markdown("""
This application predicts crop prices based on environmental parameters using machine learning.
Upload your data or use our sample dataset to explore and predict crop prices.
""")
# Add download button for the full dataset
if st.sidebar.button("Download Full Dataset"):
    with open('full_dataset.csv', 'r') as f:
        csv_data = f.read()
    st.sidebar.download_button(
        label="Click to Download",
        data=csv_data,
        file_name="crop_price_prediction_dataset.csv",
        mime="text/csv"
    )

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Data Analysis", "Model Training", "Price Prediction"])

# Load or upload data
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose data source:", ["Upload Data", "Use Sample Data"])

if data_source == "Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_sample_data()
else:
    df = load_sample_data()

if page == "Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    # Display basic statistics
    st.subheader("Dataset Overview")
    st.write(df.describe())
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    fig = px.imshow(df.corr(), 
                    color_continuous_scale='RdBu',
                    title="Correlation Heatmap")
    st.plotly_chart(fig)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature to visualize:", 
                          df.columns.tolist())
    fig = px.histogram(df, x=feature, 
                       title=f"Distribution of {feature}")
    st.plotly_chart(fig)
    
    # Scatter plots
    st.subheader("Feature Relationships")
    x_axis = st.selectbox("Select X-axis:", df.columns.tolist())
    y_axis = st.selectbox("Select Y-axis:", 
                         [col for col in df.columns if col != x_axis])
    fig = px.scatter(df, x=x_axis, y=y_axis,
                    title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig)

elif page == "Model Training":
    st.header("🤖 Model Training")
    
    # Model parameters
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of trees", 50, 500, 100)
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
    
    if st.button("Train Model"):
        # Preprocess data and train model
        X, y = preprocess_data(df)
        metrics, feature_importance = train_model(X, y, n_estimators, test_size)
        
        # Display metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{metrics['r2']:.3f}")
        col2.metric("MAE", f"{metrics['mae']:.3f}")
        col3.metric("RMSE", f"{metrics['rmse']:.3f}")
        
        # Feature importance plot
        st.subheader("Feature Importance")
        fig = px.bar(x=feature_importance.index, 
                    y=feature_importance.values,
                    title="Feature Importance")
        st.plotly_chart(fig)

else:  # Price Prediction page
    st.header("💰 Price Prediction")
    
    # Input form for predictions
    st.subheader("Enter Environmental Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        air_humidity = st.slider("Air Humidity (%)", 0, 100, 50)
        temperature = st.slider("Temperature (°C)", 0, 50, 25)
        soil_humidity = st.slider("Soil Humidity (%)", 0, 100, 50)
    
    with col2:
        soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
        rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
    
    if st.button("Predict Price"):
        # Create input data
        input_data = pd.DataFrame({
            'air_humidity': [air_humidity],
            'temperature': [temperature],
            'soil_humidity': [soil_humidity],
            'soil_ph': [soil_ph],
            'rainfall': [rainfall]
        })
        
        # Make prediction
        X, _ = preprocess_data(df)  # Get scaler fit on training data
        predicted_price = predict_price(input_data)
        
        # Display prediction
        st.success(f"Predicted Crop Price: ${predicted_price:.2f}")

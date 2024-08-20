import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Title of the web app
st.title("SBI Stock Price Analysis and Forecasting")

# Load the data with correct date format
df = pd.read_csv('C:/Users/msi00/OneDrive/Desktop/SBI.csv', parse_dates=['Date'], index_col='Date', dayfirst=True)

# Ensure the Date column is parsed correctly and the index is datetime
df.index = pd.to_datetime(df.index)

# Display the raw data
st.subheader("Raw Data")
st.write(df.head())

# Plot the closing prices
st.subheader("Closing Price Over Time")
st.line_chart(df['Close'])

# ARIMA model training
st.subheader("ARIMA Model Training")

# Allow user to specify ARIMA order
p = st.number_input("ARIMA p parameter", min_value=0, value=1, step=1)
d = st.number_input("ARIMA d parameter", min_value=0, value=1, step=1)
q = st.number_input("ARIMA q parameter", min_value=0, value=1, step=1)

# Train the ARIMA model
if st.button("Train ARIMA Model"):
    model = ARIMA(df['Close'], order=(p,d,q))
    model_fit = model.fit()
    st.write(model_fit.summary())
    
    # Forecasting
    forecast_period = st.slider("Select forecast period (days)", min_value=1, max_value=365, value=30)
    forecast = model_fit.forecast(steps=forecast_period)
    
    # Plot forecast
    st.subheader(f"Forecast for the next {forecast_period} days")
    
    # Generate a new date range for the forecast period
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_period+1, freq='D')[1:]

    # Ensure that both actual data and forecasted data have datetime indexes
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Close'], label='Actual Prices')
    plt.plot(forecast_index, forecast, label='Forecasted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('SBI Stock Price Forecast')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

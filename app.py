import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Title of the web app
st.title("Stock Price Prediction using ARIMA")

# Load the dataset directly (assuming file path is provided)
file_path = "C:/users/msi00/OneDrive/Desktop/SBI.csv"
data = pd.read_csv(file_path)

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data.set_index('Date', inplace=True)

# Display first few rows of the dataset
st.write("First few rows of the dataset:")
st.dataframe(data.head())

# Display basic information about the dataset
st.write("Dataset Info:")
st.write(data.describe())

# Plot the closing price
st.subheader("Closing Price over Time")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Use the entire dataset for ARIMA model training
stock_prices = data['Close']

# ARIMA model training
st.subheader("Training the ARIMA Model")
arima_order = st.selectbox("Select ARIMA order (p, d, q)", [(5, 1, 0), (2, 1, 2), (3, 1, 0)])

model = ARIMA(stock_prices, order=arima_order)
arima_model = model.fit()

# Predict future stock prices based on user-selected date range
st.subheader("Future Stock Price Prediction")

# Extract the last date in the stock_prices index
last_date = stock_prices.index[-1]

# Convert last_date to datetime.date for compatibility with the date input widget
last_date_as_date = last_date.date()

# Date pickers for future prediction range
start_date = st.date_input("Select the start date for prediction", value=last_date_as_date + pd.DateOffset(1), min_value=last_date_as_date + pd.DateOffset(1))
end_date = st.date_input("Select the end date for prediction", value=last_date_as_date + pd.DateOffset(30), min_value=start_date)

# Calculate the number of future steps (days) to predict based on the selected date range
future_steps = (end_date - start_date).days + 1

if future_steps > 0:
    # Re-train the ARIMA model to include future steps
    model_extended = ARIMA(stock_prices, order=arima_order)
    arima_model_extended = model_extended.fit()
    
    # Extend index to include future dates
    extended_dates = pd.date_range(start=stock_prices.index[0], end=end_date)
    
    # Forecast including in-sample and out-of-sample (future) predictions
    forecast = arima_model_extended.get_forecast(steps=len(extended_dates))
    forecast_ci = forecast.conf_int(alpha=0.05)
    
    # Extract the mean prediction and confidence intervals
    forecast_mean = forecast.predicted_mean
    forecast_lower = forecast_ci.iloc[:, 0]
    forecast_upper = forecast_ci.iloc[:, 1]
    
    # Show predicted future values
    st.write("Future Predictions:")
    future_df = pd.DataFrame({
        'Date': extended_dates[-future_steps:], 
        'Predicted Prices': forecast_mean[-future_steps:], 
        'Lower CI': forecast_lower[-future_steps:], 
        'Upper CI': forecast_upper[-future_steps:]
    })
    st.dataframe(future_df)
    
    # Plot future predictions with confidence intervals
    st.subheader("Predicted Future Stock Prices with Confidence Intervals")
    fig3, ax3 = plt.subplots()
    ax3.plot(stock_prices.index, stock_prices, label='Historical Prices', color='green')
    ax3.plot(extended_dates[-future_steps:], forecast_mean[-future_steps:], label='Future Predicted Prices', linestyle='--', color='black')
    ax3.fill_between(extended_dates[-future_steps:], forecast_lower[-future_steps:], forecast_upper[-future_steps:], color='lightblue', alpha=0.5, label='95% Confidence Interval')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price')
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax3.legend()
    st.pyplot(fig3)

    # Plot combined graph: historical and future predictions with confidence intervals
    st.subheader("Combined Graph: Historical and Future Predictions with Confidence Intervals")
    fig4, ax4 = plt.subplots()
    ax4.plot(stock_prices.index, stock_prices, label='Historical Prices', color='purple')
    ax4.plot(extended_dates[-future_steps:], forecast_mean[-future_steps:], label='Future Predicted Prices', linestyle='--', color='red')
    ax4.fill_between(extended_dates[-future_steps:], forecast_lower[-future_steps:], forecast_upper[-future_steps:], color='lightgreen', alpha=0.5, label='95% Confidence Interval')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Price')
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax4.legend()
    st.pyplot(fig4)

    # Additional: Predict the next 30 days regardless of user-selected dates
    st.subheader("Predicted Prices for the Next 30 Days")

    # Predict the next 30 days
    next_30_days = arima_model_extended.get_forecast(steps=30)
    next_30_mean = next_30_days.predicted_mean
    next_30_ci = next_30_days.conf_int(alpha=0.05)
    
    next_30_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30, freq='D')
    
    # Show the next 30 days predictions first
    st.write("Next 30 Days Predictions:")
    next_30_df = pd.DataFrame({
        'Date': next_30_dates, 
        'Predicted Prices': next_30_mean, 
        'Lower CI': next_30_ci.iloc[:, 0], 
        'Upper CI': next_30_ci.iloc[:, 1]
    })
    st.dataframe(next_30_df)

    # Plot the next 30 days predictions with confidence intervals
    fig5, ax5 = plt.subplots()
    ax5.plot(stock_prices.index, stock_prices, label='Historical Prices', color='blue')
    ax5.plot(next_30_dates, next_30_mean, label='Next 30 Days Predictions', linestyle='--', color='green')
    ax5.fill_between(next_30_dates, next_30_ci.iloc[:, 0], next_30_ci.iloc[:, 1], color='lightgreen', alpha=0.5, label='95% Confidence Interval')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Price')
    ax5.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    ax5.legend()
    st.pyplot(fig5)

else:
    st.error("Please ensure that the end date is after the start date.")

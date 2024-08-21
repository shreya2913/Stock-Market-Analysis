import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to load data
@st.cache
def load_data():
    sbi_data = pd.read_csv('C:/Users/msi00/OneDrive/Desktop/SBI.csv')
    return sbi_data

# Function to plot the data
def plot_data(df, title):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Close'], label='Actual', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to train and predict using ARIMA
def arima_forecast(df, start_date, end_date, p, d, q):
    df = df[['Date', 'Close']].copy()  # Ensure 'Date' column is available
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')  # Adjust format here
    df.set_index('Date', inplace=True)
    df = df.loc[start_date:end_date]  # Filter data based on date range
    model = ARIMA(df['Close'], order=(p, d, q))  # Use user-defined ARIMA parameters
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # Forecast for 30 days ahead
    return forecast, model_fit

# Function to train and predict using Prophet
def prophet_forecast(df, start_date, end_date):
    if df.index.name == 'Date':
        df = df.reset_index()  # Make 'Date' a column if it's an index
    df = df[['Date', 'Close']].copy()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y')  # Adjust format here
    df = df[(df['ds'] >= pd.to_datetime(start_date)) & (df['ds'] <= pd.to_datetime(end_date))]
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Load data
sbi_data = load_data()

# Ensure 'Date' column is in datetime format
sbi_data['Date'] = pd.to_datetime(sbi_data['Date'], format='%d-%m-%Y')

# Streamlit app
st.title('SBI Stock Analysis and Forecasting')

# Date range selection
min_date = sbi_data['Date'].min().date()
max_date = sbi_data['Date'].max().date()
start_date = st.date_input('Start Date', min_date)
end_date = st.date_input('End Date', max_date)

# Filter data based on selected date range
sbi_data = sbi_data[(sbi_data['Date'] >= pd.to_datetime(start_date)) & (sbi_data['Date'] <= pd.to_datetime(end_date))]
sbi_data.set_index('Date', inplace=True)  # Set 'Date' as index for filtering

# Show the actual data
st.subheader('Actual Stock Data')
plot_data(sbi_data, 'SBI Stock Prices')

# Model selection
model_option = st.selectbox('Choose Forecasting Model:', ['ARIMA', 'Prophet'])

if model_option == 'ARIMA':
    # ARIMA parameter inputs
    p = st.number_input('ARIMA Parameter p:', min_value=0, max_value=10, value=5)
    d = st.number_input('ARIMA Parameter d:', min_value=0, max_value=3, value=1)
    q = st.number_input('ARIMA Parameter q:', min_value=0, max_value=10, value=0)
    
    # Reset index before passing to the function
    sbi_data = sbi_data.reset_index()
    forecast, model_fit = arima_forecast(sbi_data, start_date, end_date, p, d, q)
    st.subheader('ARIMA Forecast')
    st.line_chart(forecast)
    st.write('Forecast Mean Squared Error:', mean_squared_error(sbi_data['Close'].iloc[-30:], forecast))
elif model_option == 'Prophet':
    forecast = prophet_forecast(sbi_data, start_date, end_date)
    st.subheader('Prophet Forecast')
    fig = plt.figure(figsize=(10, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Prophet Forecast')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set Streamlit page config
st.set_page_config(page_title="Stock Price Prediction with LSTM", layout="wide", page_icon="📈")

# Function to preprocess the stock data
def preprocess_data(df):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index(ascending=True)
    
    # Create features
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    df['Dayofyear'] = df.index.dayofyear
    df['Close_rolling_mean'] = df['Close'].rolling(window=7).mean()
    df['Close_rolling_std'] = df['Close'].rolling(window=7).std()

    return df

# Function to scale data
def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])
    return df_scaled, scaler

# Function to create LSTM data format (X, y)
def create_lstm_data(df_scaled, time_step=60):
    X, y = [], []
    for i in range(time_step, len(df_scaled)):
        X.append(df_scaled[i-time_step:i, 0])  # previous 'time_step' prices
        y.append(df_scaled[i, 0])  # the current price
    return np.array(X), np.array(y)

# Function to train LSTM model
def train_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model

# Function to plot stock price predictions
def plot_stock_price(df, y_test, y_pred, title="Stock Price Prediction"):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Actual", line=dict(color='royalblue', width=3)))
    fig.add_trace(go.Scatter(x=df.index[-len(y_test):], y=y_pred, mode='lines', name="Predicted", line=dict(color='orange', width=3, dash='dot')))

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Stock Price", template="plotly_dark")
    st.plotly_chart(fig)

# Streamlit UI
st.title("Stock Price Prediction with LSTM")

st.markdown("""
    **Upload a CSV file containing stock data. The file should include at least the 'Date' and 'Close' columns.**
""")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display data preview
    st.subheader("Data Preview")
    st.write(df.head())

    # Preprocess the data
    df = preprocess_data(df)

    # Scale the data
    df_scaled, scaler = scale_data(df)

    # Create LSTM data format
    time_step = 60  # Use previous 60 days to predict the next day's price
    X, y = create_lstm_data(df_scaled, time_step)
    
    # Reshape X to be 3D as required by LSTM (samples, time_step, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the LSTM model
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)  # Inverse scale the predicted values
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))  # Inverse scale the actual values

    # Display model performance
    mae = mean_absolute_error(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)

    st.subheader("Model Performance Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Visualize the predictions
    st.subheader("Stock Price Prediction Visualization")
    plot_stock_price(df, y_test_actual, y_pred)

    # Display LSTM Model Architecture
    st.subheader("LSTM Model Architecture")
    st.write(model.summary())

    # Closing remarks
    st.markdown("""
    This tool provides an easy way to predict stock prices using LSTM, a powerful deep learning technique for time-series forecasting. 
    You can upload your CSV file, and the app will show the model's predictions and performance.
    """)


from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tensorflow.keras.initializers import Orthogonal

app = Flask(__name__)

# Custom objects
custom_objects = {
    'Orthogonal': Orthogonal
}

# Load pre-trained models with custom objects
def load_model_with_custom_objects(model_path, custom_objects):
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except ValueError as e:
        print(f"Error loading model {model_path}: {e}")
        return None

models = {
    "AAPL": load_model_with_custom_objects("models/AAPL_model.keras", custom_objects),
    "MSFT": load_model_with_custom_objects("models/MSFT_model.keras", custom_objects),
    "GOOG": load_model_with_custom_objects("models/GOOG_model.keras", custom_objects)
}

# Function to get and preprocess stock data
def get_stock_data(ticker):
    df = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    return df

# Function to prepare data for model prediction
def prepare_data(df):
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df.values)
    X_test = []
    for i in range(100, data.shape[0]):
        X_test.append(data[i-100:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return scaler, X_test

# Function to predict next n days
def predict_future(model, data, scaler, n_days=5):
    hist = 100
    future_preds = []
    current_data = data[-hist:, 0]  # Ensuring correct slicing
    current_data = current_data.reshape((1, hist, 1))

    for _ in range(n_days):
        future_pred = model.predict(current_data)
        future_preds.append(future_pred[0, 0])
        current_data = np.append(current_data[0], future_pred)[-hist:]
        current_data = current_data.reshape((1, hist, 1))

    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    return future_preds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    if ticker not in models or models[ticker] is None:
        return jsonify({"error": "Model for the selected ticker not found"}), 400

    # Get and preprocess data
    df = get_stock_data(ticker)
    scaler, X_test = prepare_data(df)
    model = models[ticker]

    # Predict next 5 days
    future_preds = predict_future(model, X_test, scaler, n_days=5)

    # Prepare response
    response = {
        "ticker": ticker,
        "predictions": [float(pred) for pred in future_preds]
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

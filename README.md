# Stock Price Prediction App

This is a Flask web application that predicts future stock prices using pre-trained machine learning models. The application leverages Yahoo Finance to fetch historical stock data and TensorFlow Keras models to make future price predictions.

## Features

- Fetches historical stock data from Yahoo Finance.
- Uses pre-trained Keras models to predict future stock prices.
- Predicts stock prices for the next 5 days.
- Supports predictions for AAPL, MSFT, and GOOG tickers.

## Installation

### Prerequisites

- Python 3.7+
- pip

### Clone the repository

```sh
git clone https://github.com/FeverCode/Stock-Market-Prediction.git
```
```
cd Stock-Market-Prediction
```

### Install dependecies
    
 ```bash
pip install -r requirements.txt
 ```

### Run the application

Running the app
1. Ensure that you have placed the pre-trained Keras models in the models directory.
2. Run the Flask application:

```sh
python app.py
```

3. Open `http://127.0.0.1:5000/.`

### Making Predictions
1. Select the ticker symbol (AAPL, MSFT, or GOOG) from the dropdown menu.
2. Click the "Predict" button.
3. The app will display the predicted stock prices for the next 5 days.

### Acknowledgments
* [Yahoo Finance](https://finance.yahoo.com/) for providing historical stock data.
* [TensorFlow](https://www.tensorflow.org/) for the machine learning framework.






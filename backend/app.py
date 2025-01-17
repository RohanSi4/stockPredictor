from flask import Flask, request, jsonify
from flask_cors import CORS
from data_fetcher import StockDataFetcher
from model_trainer import train_lstm_model
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route('/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        # Initialize fetcher
        fetcher = StockDataFetcher(API_KEY)
        
        # Fetch raw data first
        raw_df = fetcher.fetch_stock_data(symbol)
        if raw_df is None:
            return jsonify({'error': f'No data found for {symbol}'}), 404
            
        # Get the raw close price
        raw_price = float(raw_df['4. close'].iloc[-1])
        
        # Get processed data for model
        df = fetcher.process_stock_data(symbol)
        
        # Train model and get prediction
        try:
            predicted_price = float(train_lstm_model(df, symbol))
            if np.isnan(predicted_price):
                predicted_price = raw_price  # Fallback to current price if prediction fails
        except:
            predicted_price = raw_price  # Fallback to current price if model fails
        
        # Calculate return
        potential_return = ((predicted_price - raw_price) / raw_price * 100)
        
        response_data = {
            'symbol': symbol,
            'current_price': round(raw_price, 2),
            'predicted_price': round(predicted_price, 2),
            'recommendation': "BUY" if predicted_price > raw_price else "SELL",
            'potential_return': round(potential_return, 2)
        }
        
        print("Response data:", response_data)
        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            'symbol': symbol,
            'error': str(e),
            'current_price': 0.0,
            'predicted_price': 0.0,
            'recommendation': "HOLD",
            'potential_return': 0.0
        })

if __name__ == '__main__':
    app.run(port=5001, debug=True) 
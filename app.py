from flask import Flask, request, jsonify
from flask_cors import CORS
from data_fetcher import StockDataFetcher
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize StockDataFetcher
fetcher = StockDataFetcher(API_KEY)

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/fetch-stock-data', methods=['GET'])
def fetch_stock_data():
    """Endpoint to fetch and process stock data"""
    try:
        # Get parameters from request
        symbol = request.args.get('symbol', 'SPY')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Validate symbol
        if not symbol:
            return jsonify({"error": "Symbol is required"}), 400
            
        # Process the data
        df = fetcher.process_stock_data(symbol, start_date, end_date)
        
        if df is None:
            return jsonify({"error": f"Failed to process data for {symbol}"}), 500
            
        # Convert DataFrame to dict for JSON response
        data = df.reset_index().to_dict(orient='records')
        
        return jsonify({
            "symbol": symbol,
            "data": data,
            "start_date": start_date,
            "end_date": end_date,
            "record_count": len(data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/available-symbols', methods=['GET'])
def get_available_symbols():
    """Return list of available symbols from stock_data directory"""
    try:
        symbols = []
        if os.path.exists('stock_data'):
            files = os.listdir('stock_data')
            symbols = list(set([f.split('_')[0] for f in files if f.endswith('_all_data.csv')]))
        
        return jsonify({
            "symbols": symbols,
            "count": len(symbols)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-data/<symbol>', methods=['GET'])
def get_latest_data(symbol):
    """Get the latest processed data for a symbol"""
    try:
        file_path = f'stock_data/{symbol}_processed_data.csv'
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"No data available for {symbol}"}), 404
            
        df = pd.read_csv(file_path)
        latest_data = df.iloc[-1].to_dict()  # Get the most recent record
        
        return jsonify({
            "symbol": symbol,
            "latest_data": latest_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Create stock_data directory if it doesn't exist
    os.makedirs('stock_data', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True) 
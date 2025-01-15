import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from datetime import datetime, timedelta
import numpy as np
import time

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class StockDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        # Create stock_data directory if it doesn't exist
        os.makedirs('stock_data', exist_ok=True)

    def fetch_stock_data(self, symbol, start_date=None, end_date=None):
        """Fetch stock data for any symbol"""
        try:
            print(f"Fetching data for {symbol}...")
            
            # Fetch daily data
            data, meta_data = self.ts.get_daily(symbol=symbol, outputsize='full')
            
            # Rename columns to match our format
            data.columns = ['1. open', '2. high', '3. low', '4. close', '5. volume']
            
            # Convert index to datetime if it's not already
            data.index = pd.to_datetime(data.index)
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]
            
            print(f"Retrieved {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def process_stock_data(self, symbol, start_date=None, end_date=None):
        """Process stock data for any symbol"""
        try:
            # Generate file paths based on symbol
            raw_file = f'stock_data/{symbol}_all_data.csv'
            processed_file = f'stock_data/{symbol}_processed_data.csv'
            
            print(f"Processing data for {symbol}...")
            
            # Fetch raw data
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if df is None or df.empty:
                raise Exception(f"No data retrieved for {symbol}")
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Calculate log returns
            df['log_return'] = np.log(df['4. close'] / df['4. close'].shift(1))
            
            # Calculate trend (can be customized based on your strategy)
            df['trend'] = df['4. close'].diff().apply(lambda x: 1 if x > 0 else 0)
            
            # Normalize features
            df_normalized = self.normalize_features(df)
            
            # Save both versions
            df.to_csv(raw_file)
            df_normalized.to_csv(processed_file)
            
            print(f"Data processed and saved for {symbol}")
            return df_normalized
            
        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the dataset"""
        # RSI
        delta = df['4. close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['SMA_20'] = df['4. close'].rolling(window=20).mean()
        df['EMA_20'] = df['4. close'].ewm(span=20).mean()
        
        # Bollinger Bands
        df['BB_Mid'] = df['SMA_20']
        std = df['4. close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (std * 2)
        
        # ATR
        high_low = df['2. high'] - df['3. low']
        high_close = abs(df['2. high'] - df['4. close'].shift())
        low_close = abs(df['3. low'] - df['4. close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        return df

    def normalize_features(self, df):
        """Normalize features to range [0,1]"""
        df_normalized = df.copy()
        
        # Define columns to normalize
        price_cols = ['1. open', '2. high', '3. low', '4. close']
        technical_cols = ['RSI', 'SMA_20', 'EMA_20', 'BB_Lower', 'BB_Mid', 'BB_Upper', 'ATR']
        
        # Print pre-normalization stats
        print("\nPre-normalization ranges:")
        for col in price_cols + technical_cols:
            if col in df.columns:
                print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

        # Normalize price columns
        for col in price_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)

        # Normalize technical indicators
        for col in technical_cols:
            if col == 'RSI':  # RSI is already 0-100
                df_normalized[col] = df[col] / 100
            elif col == 'ATR':
                df_normalized[col] = (np.log1p(df[col]) - np.log1p(df[col]).min()) / \
                                   (np.log1p(df[col]).max() - np.log1p(df[col]).min())
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df[col] - min_val) / (max_val - min_val)

        return df_normalized

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    fetcher = StockDataFetcher(api_key)
    
    # Example: Fetch data for multiple symbols
    symbols = ['SPY', 'AAPL', 'GOOGL']
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        df = fetcher.process_stock_data(symbol)
        if df is not None:
            print(f"Successfully processed {symbol}")
            print(f"Data shape: {df.shape}")

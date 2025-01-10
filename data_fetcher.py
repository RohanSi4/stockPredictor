import os
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
from alpha_vantage.timeseries import TimeSeries

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Create output folder
output_folder = "stock_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Fetch SPY data
print("Fetching Daily Data for SPY...")
ts = TimeSeries(key=API_KEY, output_format='pandas')
data, meta_data = ts.get_daily(symbol='SPY', outputsize='full')
data.index.name = 'date'
data.sort_index(inplace=True)

# Add technical indicators
print("Adding Technical Indicators...")
data['log_return'] = ta.log_return(data['4. close'])
data['RSI'] = ta.rsi(data['4. close'], length=14)
data['SMA_20'] = ta.sma(data['4. close'], length=20)
data['EMA_20'] = ta.ema(data['4. close'], length=20)
bbands = ta.bbands(data['4. close'], length=20)  # Adjust length if needed
data['BB_Lower'] = bbands['BBL_20_2.0']  # Lower Band
data['BB_Mid'] = bbands['BBM_20_2.0']    # Middle Band
data['BB_Upper'] = bbands['BBU_20_2.0']  # Upper Band
data['ATR'] = ta.atr(data['2. high'], data['3. low'], data['4. close'], length=14)
data['trend'] = (data['4. close'] - ta.sma(data['4. close'], length=50)) / ta.sma(data['4. close'], length=50)

# Feature engineering
data['target'] = (data['4. close'].shift(-1) > data['4. close']).astype(int)

# Drop missing values and unnecessary columns
data.dropna(inplace=True)
data.drop(columns=['5. volume'], inplace=True)

# Save processed data
raw_file_path = os.path.join(output_folder, "SPY_all_data.csv")
processed_file_path = os.path.join(output_folder, "SPY_processed_data.csv")
data.to_csv(raw_file_path)
data.to_csv(processed_file_path)
print(f"Raw data saved to {raw_file_path}")
print(f"Processed data saved to {processed_file_path}")

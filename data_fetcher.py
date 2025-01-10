import os
from dotenv import load_dotenv
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Alpha Vantage API setup
output_folder = "stock_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
file_path = os.path.join(output_folder, "SPY_all_data.csv")

# Fetch all available daily data
print("Fetching full historical data for SPY...")
ts = TimeSeries(key=API_KEY, output_format='pandas')
data, meta_data = ts.get_daily(symbol='SPY', outputsize='full')
data.to_csv(file_path, index=True)
print(f"Data saved to {file_path}")

# Load data for processing
data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
data.sort_index(inplace=True)

# Feature engineering
data['daily_change'] = (data['4. close'] - data['1. open']) / data['1. open']
data['target'] = (data['4. close'] > data['1. open']).astype(int)  # 1 if Close > Open, else 0

# Drop unnecessary columns
data.drop(columns=['5. volume'], inplace=True)
data.dropna(inplace=True)

# Save processed data to disk
processed_file_path = os.path.join(output_folder, "SPY_processed_data.csv")
data.to_csv(processed_file_path)
print(f"Processed data saved to {processed_file_path}")

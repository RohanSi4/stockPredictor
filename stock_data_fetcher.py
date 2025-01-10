import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
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

# Chronological train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

X_train = train_data[['1. open', '2. high', '3. low', '4. close', 'daily_change']]
y_train = train_data['target']
X_test = test_data[['1. open', '2. high', '3. low', '4. close', 'daily_change']]
y_test = test_data['target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model evaluation function
def evaluate_model(model, model_name):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, y_pred)
    print(f"{model_name} AUC-ROC Score: {auc:.4f}")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))
    return auc

# Logistic Regression
log_reg = LogisticRegression()
evaluate_model(log_reg, "Logistic Regression")

# Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
evaluate_model(decision_tree, "Decision Tree")

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(random_forest, "Random Forest")

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
evaluate_model(xgb, "XGBoost")

# Support Vector Machine
svm = SVC(probability=True, random_state=42)
evaluate_model(svm, "SVM")

# LSTM Model
sequence_length = 30  # Use 30-day sequences
batch_size = 64

# Prepare data for LSTM
train_generator = TimeseriesGenerator(X_train_scaled, y_train, length=sequence_length, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test_scaled, y_test, length=sequence_length, batch_size=batch_size)

# Build the LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train_scaled.shape[1])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
lstm_model.fit(train_generator, epochs=20, verbose=1)

# Evaluate LSTM
lstm_auc = lstm_model.evaluate(test_generator, verbose=0)[1]
print(f"LSTM AUC-ROC Score: {lstm_auc:.4f}")

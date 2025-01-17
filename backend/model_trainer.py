import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_lstm_model(data, symbol):
    # Use only available features
    features = ['4. close', 'RSI', 'SMA_20', 'EMA_20', 'BB_Lower', 'BB_Mid', 'BB_Upper', 'ATR']
    X = data[features]
    
    # Prepare target (next day's close price)
    y = data['4. close'].shift(-1)
    
    # Remove last row since we don't have target for it
    X = X[:-1]
    y = y[:-1].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    sequence_length = 10
    X_sequences = []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:(i + sequence_length)])
    X_sequences = np.array(X_sequences)
    
    # Prepare target values
    y = y[sequence_length:]
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, len(features)), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X_sequences, y, epochs=50, batch_size=32, verbose=0)
    
    # Prepare last sequence for prediction
    last_sequence = X_scaled[-sequence_length:]
    last_sequence = np.reshape(last_sequence, (1, sequence_length, len(features)))
    
    # Make prediction
    predicted_scaled = model.predict(last_sequence)
    
    # Get the last known price
    last_price = data['4. close'].iloc[-1]
    
    # Calculate predicted price (assuming relative change)
    predicted_change = predicted_scaled[0][0]
    predicted_price = last_price * (1 + predicted_change)
    
    return predicted_price

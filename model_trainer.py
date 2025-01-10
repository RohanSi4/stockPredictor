import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load processed data
processed_file_path = "stock_data/SPY_processed_data.csv"
data = pd.read_csv(processed_file_path, parse_dates=['date'], index_col='date')

# Features and target
X = data[['1. open', '2. high', '3. low', '4. close', 'log_return', 'RSI', 'SMA_20', 'EMA_20', 'trend', 'BB_Lower', 'BB_Mid', 'BB_Upper', 'ATR']]
y = data['target']

# Chronological train-test split
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Model evaluation function
def evaluate_model(model, model_name):
    model.fit(X_train_scaled, y_train_balanced)
    y_pred = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n{model_name} AUC-ROC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return auc

# Logistic Regression with Hyperparameter Tuning
param_grid_log = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_log = GridSearchCV(LogisticRegression(random_state=42), param_grid_log, cv=3, scoring='roc_auc')
evaluate_model(grid_log, "Logistic Regression (Tuned)")

# Random Forest with Hyperparameter Tuning
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='roc_auc')
evaluate_model(grid_rf, "Random Forest (Tuned)")

# SVM with Hyperparameter Tuning
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_svm = GridSearchCV(SVC(probability=True, random_state=42), param_grid_svm, cv=3, scoring='roc_auc')
evaluate_model(grid_svm, "SVM (Tuned)")

# LSTM Model
sequence_length = 30
batch_size = 64

# Prepare data for LSTM
train_generator = TimeseriesGenerator(X_train_scaled, y_train_balanced, length=sequence_length, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test_scaled, y_test, length=sequence_length, batch_size=batch_size)

# Build and train LSTM
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
print(f"\nLSTM AUC-ROC Score: {lstm_auc:.4f}")

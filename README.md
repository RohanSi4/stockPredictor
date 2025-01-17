# AI Stock Price Predictor

A full-stack web application that predicts stock prices using machine learning algorithms and technical indicators. Built with React, Flask, and TensorFlow.

## Features
- Real-time stock data fetching using Alpha Vantage API
- Technical analysis with RSI, SMA, EMA, and Bollinger Bands
- LSTM-based price prediction model
- Interactive web interface
- Buy/Sell recommendations based on predictions

## Prerequisites
- Python 3.8+
- Node.js 14+
- Alpha Vantage API key (free at https://www.alphavantage.co/support/#api-key)

## Installation

### Backend Setup
1. Clone the repository
```bash
git clone https://github.com/YourUsername/stock-predictor.git
cd stock-predictor
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

4. Create a .env file in the backend directory
```bash
cd backend
touch .env  # On Windows: type nul > .env
```

5. Add your Alpha Vantage API key to the .env file:
```plaintext
ALPHA_VANTAGE_API_KEY=your_api_key_here
FLASK_APP=app.py
FLASK_ENV=development
PORT=5001
```

Note: Replace 'your_api_key_here' with your actual Alpha Vantage API key. You can get a free API key at https://www.alphavantage.co/support/#api-key

Important: Never commit your .env file to version control. The .gitignore file should include .env to keep your API key private.

### Frontend Setup
1. Install Node dependencies
```bash
cd ../frontend
npm install
```

## Running the Application

1. Start the Flask backend server
```bash
cd backend
python app.py
```
The backend will run on http://localhost:5001

2. In a new terminal, start the React frontend
```bash
cd frontend
npm start
```
The frontend will run on http://localhost:3000

## Usage
1. Open http://localhost:3000 in your browser
2. Enter a stock symbol (e.g., AAPL, GOOGL, SPY)
3. Click "Predict" to get:
   - Current price
   - Predicted price
   - Buy/Sell recommendation
   - Potential return percentage

## Project Structure
```
stock-predictor/
├── backend/
│   ├── app.py                 # Flask server
│   ├── data_fetcher.py        # Stock data retrieval
│   ├── model_trainer.py       # LSTM model
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.js            # Main app component
│   │   └── index.js          # Entry point
│   └── package.json          # Node dependencies
└── README.md
```

## Dependencies
### Backend
- Flask
- pandas
- numpy
- tensorflow
- scikit-learn
- alpha_vantage
- python-dotenv

### Frontend
- React
- Material-UI
- Axios

## Common Issues
1. **API Rate Limits**: Alpha Vantage has a rate limit of 5 API calls per minute for free tier
2. **Model Training Time**: Initial prediction might take longer due to model training
3. **CORS Issues**: Make sure both frontend and backend are running on specified ports

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments
- Alpha Vantage for stock data API
- TensorFlow team for ML framework
- React team for frontend framework 
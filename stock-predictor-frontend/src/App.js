import { useState } from 'react';
import './App.css';

function App() {
  const [stockSymbol, setStockSymbol] = useState('');
  const [currentPrice, setCurrentPrice] = useState('0.00');
  const [predictedPrice, setPredictedPrice] = useState('0.00');

  const handlePredict = async () => {
    if (!stockSymbol) return;

    try {
      // Replace with your actual API endpoint
      const response = await fetch(`http://localhost:5001/predict/${stockSymbol}`);
      const data = await response.json();
      
      if (data) {
        setCurrentPrice(data.current_price?.toFixed(2) || '0.00');
        setPredictedPrice(data.predicted_price?.toFixed(2) || '0.00');
      }
    } catch (error) {
      console.error('Error fetching prediction:', error);
      // Optionally add error handling UI here
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Price Predictor</h1>
      </header>
      <main className="App-main">
        <div className="stock-input-section">
          <input 
            type="text" 
            value={stockSymbol}
            onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
            placeholder="Enter stock symbol (e.g., AAPL)"
          />
          <button onClick={handlePredict}>Predict</button>
        </div>
        
        <div className="stock-display-section">
          <div className="current-price">
            <h2>Current Price</h2>
            <p>${currentPrice}</p>
          </div>
          
          <div className="prediction">
            <h2>Predicted Price</h2>
            <p>${predictedPrice}</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

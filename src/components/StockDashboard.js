import React, { useState } from 'react';

function StockDashboard() {
  const [stockSymbol, setStockSymbol] = useState('');
  const [currentPrice, setCurrentPrice] = useState('0.00');
  const [predictedPrice, setPredictedPrice] = useState('0.00');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async () => {
    if (!stockSymbol) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:5001/predict/${stockSymbol}`);
      const data = await response.json();
      
      if (data) {
        setCurrentPrice(data.current_price?.toFixed(2) || '0.00');
        setPredictedPrice(data.predicted_price?.toFixed(2) || '0.00');
      }
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App-main">
      <div className="stock-input-section">
        <input 
          type="text" 
          value={stockSymbol}
          onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
          placeholder="Enter stock symbol (e.g., AAPL)"
          disabled={loading}
        />
        <button 
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Predict'}
        </button>
      </div>
      
      {error && (
        <div style={{ color: '#ff6b6b', marginBottom: '20px' }}>
          Error: {error}
        </div>
      )}
      
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
    </div>
  );
}

export default StockDashboard; 
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Price Predictor</h1>
      </header>
      <main className="App-main">
        <div className="stock-input-section">
          <input 
            type="text" 
            placeholder="Enter stock symbol (e.g., AAPL)"
          />
          <button>Predict</button>
        </div>
        
        <div className="stock-display-section">
          <div className="current-price">
            <h2>Current Price</h2>
            <p>$0.00</p>
          </div>
          
          <div className="prediction">
            <h2>Predicted Price</h2>
            <p>$0.00</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

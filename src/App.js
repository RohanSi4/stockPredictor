import React from 'react';
import StockDashboard from './components/StockDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Price Predictor</h1>
      </header>
      <StockDashboard />
    </div>
  );
}

export default App; 
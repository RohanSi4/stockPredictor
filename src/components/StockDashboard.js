import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Grid,
  TextField,
  Button,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import StockChart from './StockChart';
import { fetchStockData } from '../services/api';

function StockDashboard() {
  const [symbol, setSymbol] = useState('SPY');
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchStockData(symbol);
      setStockData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleFetchData();
  }, []);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Grid container spacing={3}>
        {/* Search Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <TextField
                label="Stock Symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                sx={{ width: 200 }}
              />
              <Button
                variant="contained"
                onClick={handleFetchData}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Fetch Data'}
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Error Display */}
        {error && (
          <Grid item xs={12}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {/* Chart Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            {stockData ? (
              <StockChart data={stockData.data} />
            ) : (
              <Typography>No data available</Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

export default StockDashboard; 
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const fetchStockData = async (symbol) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/fetch-stock-data`, {
      params: { symbol },
    });
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.error || 'Failed to fetch stock data'
    );
  }
};

export const getAvailableSymbols = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/available-symbols`);
    return response.data;
  } catch (error) {
    throw new Error(
      error.response?.data?.error || 'Failed to fetch available symbols'
    );
  }
}; 
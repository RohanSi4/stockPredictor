import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

function StockChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="4. close"
          stroke="#8884d8"
          name="Close Price"
        />
        <Line
          type="monotone"
          dataKey="RSI"
          stroke="#82ca9d"
          name="RSI"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default StockChart; 
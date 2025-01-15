import React from 'react';
import { Box, Container, AppBar, Toolbar, Typography } from '@mui/material';
import StockDashboard from './StockDashboard';

function Layout() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Stock Market Predictor
          </Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <StockDashboard />
      </Container>
    </Box>
  );
}

export default Layout; 
const express = require('express');
const path = require('path');
const app = express();
const port = 5000;

// Serve static files
app.use(express.static(path.join(__dirname, 'frontend/build')));

// Simple API check endpoint
app.get('/api-check', (req, res) => {
  res.json({ message: 'React Server Running' });
});

// Root route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend/build/index.html'));
});

// Start the server
app.listen(port, () => {
  console.log(`React frontend server is running on port ${port}`);
});
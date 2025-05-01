const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Serve static files from the React frontend app
app.use(express.static(path.join(__dirname, 'frontend/build')));

// Serve our api route that returns a simple message
app.get('/api-check', (req, res) => {
  res.json({ message: 'React Server Running' });
});

// Handles any requests that don't match the ones above
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname + '/frontend/build/index.html'));
});

app.listen(port, () => {
  console.log(`React frontend server is running on port ${port}`);
});
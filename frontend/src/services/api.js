import axios from 'axios';

const API_URL = 'http://localhost:5001/api';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
const apiService = {
  // Health check
  checkHealth: () => {
    return apiClient.get('/health');
  },
  
  // Dataset operations
  uploadDataset: (formData) => {
    return apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  getDatasetInfo: () => {
    return apiClient.get('/dataset');
  },
  
  // Model operations
  getAvailableModels: () => {
    return apiClient.get('/models');
  },
  
  trainModel: (modelName, epochs = 3, batchSize = 32) => {
    return apiClient.post('/train', {
      model: modelName,
      epochs: epochs,
      batch_size: batchSize,
    });
  },
  
  evaluateModel: (modelName) => {
    return apiClient.post('/evaluate', {
      model: modelName,
    });
  },
  
  // Prediction
  analyzeText: (text, modelName = 'deberta') => {
    return apiClient.post('/predict', {
      text: text,
      model: modelName,
    });
  },
};

export default apiService;
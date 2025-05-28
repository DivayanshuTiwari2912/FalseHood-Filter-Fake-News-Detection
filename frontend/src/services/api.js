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
  
  // Train multiple models (for future extensibility if backend supports batch)
  trainMultipleModels: (models, epochs = 3, batchSize = 32) => {
    return apiClient.post('/train', {
      models: models,
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
  
  // Web scraping
  scrapeWebsite: (url) => {
    return apiClient.post('/scrape', {
      url: url,
    });
  },
  
  // Generate shareable link
  generateShareableLink: (analysisResult) => {
    // Generate shareable URL with analysis result data
    const baseUrl = window.location.origin;
    const path = '/analyze';
    
    // Create parameters for sharing
    const params = new URLSearchParams();
    if (analysisResult) {
      params.append('result', JSON.stringify({
        prediction: analysisResult.prediction,
        confidence: analysisResult.confidence,
        emoji: analysisResult.emoji,
        emoji_description: analysisResult.emoji_description,
        model: analysisResult.model
      }));
    }
    
    return `${baseUrl}${path}?${params.toString()}`;
  },
  
  // Share to social media
  shareToSocialMedia: (platform, analysisResult) => {
    const shareableLink = apiService.generateShareableLink(analysisResult);
    const shareText = `I analyzed this content with Falsehood Filter and found it to be ${analysisResult.prediction === 1 ? 'authentic' : 'false'} information (${analysisResult.emoji_description}) ${analysisResult.emoji}`;
    
    let shareUrl = '';
    
    switch (platform) {
      case 'twitter':
        shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareableLink)}`;
        break;
      case 'facebook':
        shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareableLink)}&quote=${encodeURIComponent(shareText)}`;
        break;
      case 'linkedin':
        shareUrl = `https://www.linkedin.com/shareArticle?mini=true&url=${encodeURIComponent(shareableLink)}&title=${encodeURIComponent('Falsehood Filter Analysis')}&summary=${encodeURIComponent(shareText)}`;
        break;
      case 'whatsapp':
        shareUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent(`${shareText} ${shareableLink}`)}`;
        break;
      case 'email':
        shareUrl = `mailto:?subject=${encodeURIComponent('Falsehood Filter Analysis')}&body=${encodeURIComponent(`${shareText}\n\nCheck it out here: ${shareableLink}`)}`;
        break;
      default:
        return null;
    }
    
    // Open in a new window
    if (shareUrl) {
      window.open(shareUrl, '_blank', 'noopener,noreferrer');
      return true;
    }
    
    return false;
  },
};

export default apiService;
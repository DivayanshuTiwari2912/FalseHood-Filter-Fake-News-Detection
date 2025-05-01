import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import apiService from '../services/api';
import ScrapeContent from './ScrapeContent';

const AnalyzeText = ({ trainedModels }) => {
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [activeTab, setActiveTab] = useState('manual');
  
  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await apiService.getAvailableModels();
        setAvailableModels(response.data.available_models);
        
        // Set default model if none selected
        if (!selectedModel && response.data.trained_models && response.data.trained_models.length > 0) {
          setSelectedModel(response.data.trained_models[0]);
        }
      } catch (err) {
        setError('Error fetching available models. Please try again later.');
        console.error('Error fetching models:', err);
      }
    };
    
    fetchModels();
  }, [selectedModel, trainedModels]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    if (!selectedModel) {
      setError('Please select a model first. If no models are available, please train a model in the Upload & Train section.');
      return;
    }
    
    setIsAnalyzing(true);
    setError('');
    setResult(null);
    
    try {
      const response = await apiService.analyzeText(text, selectedModel);
      setResult(response.data);
    } catch (err) {
      console.error('Error analyzing text:', err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Error analyzing text. Please try again later.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  const handleScrapedContent = (content) => {
    setText(content);
    setActiveTab('manual');
  };
  
  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.8) return 'text-success';
    if (confidence >= 0.5) return 'text-warning';
    return 'text-danger';
  };
  
  const getConfidenceBarClass = (confidence) => {
    if (confidence >= 0.8) return 'bg-success';
    if (confidence >= 0.5) return 'bg-warning';
    return 'bg-danger';
  };
  
  return (
    <div className="analyze-text-container">
      <h2 className="page-title">Analyze Text</h2>
      
      {trainedModels.length === 0 && (
        <div className="alert alert-warning">
          <p>No trained models available. Please train a model first.</p>
          <Link to="/upload-train" className="btn btn-primary">
            Go to Upload & Train
          </Link>
        </div>
      )}
      
      <div className="card mb-4">
        <div className="card-body">
          <ul className="nav nav-tabs mb-3">
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === 'manual' ? 'active' : ''}`}
                onClick={() => setActiveTab('manual')}
              >
                Manual Input
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${activeTab === 'scrape' ? 'active' : ''}`}
                onClick={() => setActiveTab('scrape')}
              >
                Web Scraper
              </button>
            </li>
          </ul>
          
          {activeTab === 'manual' ? (
            <>
              <h3 className="card-title">Enter Text to Analyze</h3>
              <form onSubmit={handleSubmit}>
                <div className="mb-3">
                  <label htmlFor="text-input" className="form-label">
                    News Article Text
                  </label>
                  <textarea
                    id="text-input"
                    className="form-control"
                    rows="6"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste news article text here..."
                  />
                </div>
                
                <div className="mb-3">
                  <label htmlFor="model-select" className="form-label">
                    Select Model
                  </label>
                  <select
                    id="model-select"
                    className="form-select"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    <option value="">Select a model</option>
                    {trainedModels.map((model) => (
                      <option key={model} value={model}>
                        {availableModels[model] || model}
                      </option>
                    ))}
                  </select>
                </div>
                
                {error && (
                  <div className="alert alert-danger" role="alert">
                    {error}
                  </div>
                )}
                
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={isAnalyzing || !text.trim() || !selectedModel}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                      &nbsp;Analyzing...
                    </>
                  ) : (
                    'Analyze Text'
                  )}
                </button>
              </form>
            </>
          ) : (
            <ScrapeContent onContentLoaded={handleScrapedContent} />
          )}
        </div>
      </div>
      
      {result && (
        <div className={`card result-card ${result.prediction === 1 ? 'real' : 'fake'}`}>
          <div className="card-body">
            <h3 className="card-title">Analysis Result</h3>
            <div className="alert alert-light">
              <h4 className={`alert-heading ${result.prediction === 1 ? 'text-success' : 'text-danger'}`}>
                <strong>
                  {result.prediction === 1 ? 'REAL NEWS' : 'FAKE NEWS'}
                </strong>
              </h4>
              <p>
                This news article appears to be{' '}
                <strong>{result.prediction === 1 ? 'genuine' : 'fake'}</strong> with a confidence of{' '}
                <span className={getConfidenceClass(result.confidence)}>
                  <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                </span>
              </p>
              
              <div className="confidence-bar mt-3">
                <div
                  className={`confidence-bar-fill ${getConfidenceBarClass(result.confidence)}`}
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
              <small className="text-muted mt-2 d-block">
                Model used: {availableModels[result.model] || result.model}
              </small>
            </div>
            
            <div className="mt-3">
              <h5>Explanation</h5>
              <p>
                {result.prediction === 1
                  ? 'The selected model has classified this content as real news based on its linguistic patterns, factual consistency, and structural characteristics that are typically associated with genuine news articles.'
                  : 'The selected model has classified this content as fake news based on detected language patterns, inconsistencies, or other characteristics that are typically associated with misleading or fabricated news articles.'}
              </p>
              <p className="text-muted">
                <small>
                  Note: This is an automated analysis and should be used as one of several tools to verify information.
                  Always cross-check important news with multiple reliable sources.
                </small>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyzeText;
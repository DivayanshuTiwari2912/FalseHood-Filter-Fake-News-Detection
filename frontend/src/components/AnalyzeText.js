import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  TwitterShareButton, FacebookShareButton, LinkedinShareButton, WhatsappShareButton, EmailShareButton,
  TwitterIcon, FacebookIcon, LinkedinIcon, WhatsappIcon, EmailIcon 
} from 'react-share';
import apiService from '../services/api';
import ScrapeContent from './ScrapeContent';
import TutorialButton from './TutorialButton';

const AnalyzeText = ({ trainedModels }) => {
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [activeTab, setActiveTab] = useState('manual');
  const location = useLocation();
  
  // Check for shared analysis result in URL
  useEffect(() => {
    // Parse shared result from URL if present
    const params = new URLSearchParams(location.search);
    const sharedResult = params.get('result');
    
    if (sharedResult) {
      try {
        const parsedResult = JSON.parse(sharedResult);
        setResult({
          ...parsedResult,
          success: true,
          label: parsedResult.prediction === 1 ? 'Authentic' : 'False'
        });
      } catch (err) {
        console.error('Error parsing shared result:', err);
      }
    }
  }, [location]);
  
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
      <TutorialButton pageName="analyze" />
      
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
              <div className="d-flex align-items-center">
                <h4 className={`alert-heading mb-0 ${result.prediction === 1 ? 'text-success' : 'text-danger'}`}>
                  <strong>
                    {result.prediction === 1 ? 'AUTHENTIC INFORMATION' : 'FALSE INFORMATION'}
                  </strong>
                </h4>
                {result.emoji && (
                  <span className="emoji-credibility ms-3" style={{ fontSize: '2rem' }}>
                    {result.emoji}
                  </span>
                )}
              </div>
              
              {result.emoji_description && (
                <div className="emoji-description mt-2 mb-3">
                  <span className="badge bg-secondary">{result.emoji_description}</span>
                </div>
              )}
              
              <p>
                This content appears to be{' '}
                <strong>{result.prediction === 1 ? 'authentic' : 'false'}</strong> with a confidence of{' '}
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
                  ? 'The selected model has classified this content as authentic information based on its linguistic patterns, factual consistency, and structural characteristics that are typically associated with reliable content.'
                  : 'The selected model has classified this content as false information based on detected language patterns, inconsistencies, or other characteristics that are typically associated with misleading or fabricated content.'}
              </p>
              <p className="text-muted">
                <small>
                  Note: This is an automated analysis and should be used as one of several tools to verify information.
                  Always cross-check important news with multiple reliable sources.
                </small>
              </p>
            </div>
            
            <div className="mt-4">
              <h5>Share this Analysis</h5>
              <div className="d-flex align-items-center share-buttons">
                <p className="me-3 mb-0">One-click sharing: </p>
                <div className="social-share-buttons">
                  <TwitterShareButton 
                    url={apiService.generateShareableLink(result)}
                    title={`I analyzed this content with Falsehood Filter and found it to be ${result.prediction === 1 ? 'authentic' : 'false'} information (${result.emoji_description}) ${result.emoji}`}
                    className="me-2"
                  >
                    <TwitterIcon size={32} round />
                  </TwitterShareButton>
                  
                  <FacebookShareButton 
                    url={apiService.generateShareableLink(result)}
                    quote={`I analyzed this content with Falsehood Filter and found it to be ${result.prediction === 1 ? 'authentic' : 'false'} information (${result.emoji_description}) ${result.emoji}`}
                    className="me-2"
                  >
                    <FacebookIcon size={32} round />
                  </FacebookShareButton>
                  
                  <LinkedinShareButton 
                    url={apiService.generateShareableLink(result)}
                    title="Falsehood Filter Analysis"
                    summary={`I analyzed this content with Falsehood Filter and found it to be ${result.prediction === 1 ? 'authentic' : 'false'} information (${result.emoji_description}) ${result.emoji}`}
                    className="me-2"
                  >
                    <LinkedinIcon size={32} round />
                  </LinkedinShareButton>
                  
                  <WhatsappShareButton 
                    url={apiService.generateShareableLink(result)}
                    title={`I analyzed this content with Falsehood Filter and found it to be ${result.prediction === 1 ? 'authentic' : 'false'} information (${result.emoji_description}) ${result.emoji}`}
                    className="me-2"
                  >
                    <WhatsappIcon size={32} round />
                  </WhatsappShareButton>
                  
                  <EmailShareButton 
                    url={apiService.generateShareableLink(result)}
                    subject="Falsehood Filter Analysis"
                    body={`I analyzed this content with Falsehood Filter and found it to be ${result.prediction === 1 ? 'authentic' : 'false'} information (${result.emoji_description}) ${result.emoji}\n\n`}
                  >
                    <EmailIcon size={32} round />
                  </EmailShareButton>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalyzeText;
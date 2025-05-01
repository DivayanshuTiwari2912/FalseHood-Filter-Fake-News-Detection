import React, { useState } from 'react';
import apiService from '../services/api';

const ScrapeContent = ({ onContentLoaded }) => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const handleScrape = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError('Please enter a URL to scrape.');
      return;
    }
    
    // Simple URL validation
    if (!url.match(/^https?:\/\/.+\..+/)) {
      setError('Please enter a valid URL. (e.g., https://example.com)');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      // Make API call to scrape content
      const response = await apiService.scrapeWebsite(url);
      
      if (response.data && response.data.text) {
        onContentLoaded(response.data.text);
      } else {
        setError('No content found or could not scrape the website.');
      }
    } catch (err) {
      console.error('Error scraping website:', err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Error scraping website. Please try a different URL or check your connection.');
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="scrape-content">
      <div className="card">
        <div className="card-body">
          <h5 className="card-title">Scrape Content from Website</h5>
          <p className="card-text">
            Enter a URL to extract content from a website for analysis.
          </p>
          
          <form onSubmit={handleScrape}>
            <div className="input-group mb-3">
              <input
                type="text"
                className="form-control"
                placeholder="https://example.com/article"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
              />
              <button
                className="btn btn-outline-primary"
                type="submit"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                    &nbsp;Scraping...
                  </>
                ) : (
                  'Scrape Content'
                )}
              </button>
            </div>
            
            {error && (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            )}
            
            <small className="text-muted">
              Note: Some websites may block web scraping attempts. News sites, blogs, and public content
              websites are most likely to work well.
            </small>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ScrapeContent;
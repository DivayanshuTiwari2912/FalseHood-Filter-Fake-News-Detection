import React, { useState, useEffect } from 'react';
import { FiGlobe, FiClock, FiBookmark, FiTrendingUp } from 'react-icons/fi';
import apiService from '../services/api';

const ScrapeContent = ({ onContentLoaded }) => {
  const [url, setUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [popularSources, setPopularSources] = useState([]);
  const [savedSources, setSavedSources] = useState([]);
  const [showTrending, setShowTrending] = useState(false);
  
  // Popular news sources
  useEffect(() => {
    // These would ideally come from an API or backend storage
    setPopularSources([
      { name: 'CNN', url: 'https://www.cnn.com' },
      { name: 'BBC', url: 'https://www.bbc.com/news' },
      { name: 'Reuters', url: 'https://www.reuters.com' },
      { name: 'The New York Times', url: 'https://www.nytimes.com' },
      { name: 'The Guardian', url: 'https://www.theguardian.com' },
      { name: 'Al Jazeera', url: 'https://www.aljazeera.com' }
    ]);
    
    // Load saved sources from localStorage
    const saved = localStorage.getItem('falsehood_filter_saved_sources');
    if (saved) {
      try {
        setSavedSources(JSON.parse(saved));
      } catch (e) {
        console.error('Error loading saved sources', e);
        setSavedSources([]);
      }
    }
  }, []);
  
  // Select a popular or saved source
  const selectSource = (sourceUrl) => {
    setUrl(sourceUrl);
  };
  
  // Save current URL to saved sources
  const saveCurrentSource = () => {
    if (!url.trim() || !url.match(/^https?:\/\/.+\..+/)) {
      setError('Please enter a valid URL before saving.');
      return;
    }
    
    // Get domain name for source name
    let sourceName = '';
    try {
      const urlObj = new URL(url);
      sourceName = urlObj.hostname.replace('www.', '');
    } catch (e) {
      sourceName = url;
    }
    
    const newSource = { name: sourceName, url: url };
    
    // Check if already saved
    if (savedSources.some(source => source.url === url)) {
      return;
    }
    
    const updatedSources = [...savedSources, newSource];
    setSavedSources(updatedSources);
    
    // Save to localStorage
    localStorage.setItem('falsehood_filter_saved_sources', JSON.stringify(updatedSources));
  };
  
  // Remove a saved source
  const removeSavedSource = (sourceUrl) => {
    const updatedSources = savedSources.filter(source => source.url !== sourceUrl);
    setSavedSources(updatedSources);
    localStorage.setItem('falsehood_filter_saved_sources', JSON.stringify(updatedSources));
  };
  
  // Handle URL scraping
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
          <h5 className="card-title">Real-Time News Scraping</h5>
          <p className="card-text">
            Enter a URL to extract content from a website for analysis, or choose from popular sources.
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
                type="button"
                className="btn btn-outline-secondary"
                onClick={saveCurrentSource}
                title="Save this source"
              >
                <FiBookmark />
              </button>
              <button
                type="submit"
                className="btn btn-outline-primary"
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
          </form>
          
          {/* Popular and saved sources */}
          <div className="mt-4">
            <ul className="nav nav-tabs" role="tablist">
              <li className="nav-item" role="presentation">
                <button 
                  className={`nav-link ${!showTrending ? 'active' : ''}`}
                  onClick={() => setShowTrending(false)}
                >
                  <FiGlobe className="me-1" /> Popular Sources
                </button>
              </li>
              <li className="nav-item" role="presentation">
                <button 
                  className={`nav-link ${showTrending ? 'active' : ''}`}
                  onClick={() => setShowTrending(true)}
                >
                  <FiTrendingUp className="me-1" /> Your Saved Sources
                </button>
              </li>
            </ul>
            
            <div className="tab-content p-3 border border-top-0 rounded-bottom">
              {!showTrending ? (
                <div className="popular-sources">
                  <div className="row row-cols-1 row-cols-md-3 g-3">
                    {popularSources.map((source, index) => (
                      <div className="col" key={index}>
                        <div className="card h-100" style={{ cursor: 'pointer' }} onClick={() => selectSource(source.url)}>
                          <div className="card-body d-flex align-items-center">
                            <FiGlobe className="me-2 text-primary" />
                            <span>{source.name}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="saved-sources">
                  {savedSources.length === 0 ? (
                    <p className="text-muted text-center py-4">
                      You don't have any saved sources yet. Click the bookmark icon to save a source.
                    </p>
                  ) : (
                    <div className="row row-cols-1 row-cols-md-3 g-3">
                      {savedSources.map((source, index) => (
                        <div className="col" key={index}>
                          <div className="card h-100">
                            <div className="card-body">
                              <div className="d-flex justify-content-between align-items-center">
                                <div 
                                  className="source-name" 
                                  style={{ cursor: 'pointer' }} 
                                  onClick={() => selectSource(source.url)}
                                >
                                  <FiBookmark className="me-2 text-success" />
                                  <span>{source.name}</span>
                                </div>
                                <button 
                                  className="btn btn-sm btn-outline-danger" 
                                  onClick={() => removeSavedSource(source.url)}
                                >
                                  &times;
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          
          <small className="text-muted mt-3 d-block">
            Note: Some websites may block web scraping attempts. News sites, blogs, and public content
            websites are most likely to work well.
          </small>
        </div>
      </div>
    </div>
  );
};

export default ScrapeContent;
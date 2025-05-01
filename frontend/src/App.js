import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from './components/Home';
import About from './components/About';
import AnalyzeText from './components/AnalyzeText';
import UploadTrain from './components/UploadTrain';
import ResultsDashboard from './components/ResultsDashboard';
import CollaborationPanel from './components/CollaborationPanel';
import apiService from './services/api';

function App() {
  const [dataset, setDataset] = useState(null);
  const [trainedModels, setTrainedModels] = useState([]);
  const [evaluationResults, setEvaluationResults] = useState({});
  const [apiStatus, setApiStatus] = useState('checking');
  
  // Check API server status
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await apiService.checkHealth();
        setApiStatus('connected');
      } catch (error) {
        console.error('Error connecting to API:', error);
        setApiStatus('disconnected');
      }
    };
    
    checkApiHealth();
    
    // Try to get dataset info if available
    const getDatasetInfo = async () => {
      try {
        const response = await apiService.getDatasetInfo();
        if (response.data && response.data.dataset) {
          setDataset(response.data.dataset);
        }
      } catch (error) {
        console.error('Error fetching dataset info:', error);
      }
    };
    
    // Try to get models info if available
    const getModelsInfo = async () => {
      try {
        const response = await apiService.getAvailableModels();
        if (response.data && response.data.trained_models) {
          setTrainedModels(response.data.trained_models);
        }
      } catch (error) {
        console.error('Error fetching models info:', error);
      }
    };
    
    if (apiStatus === 'connected') {
      getDatasetInfo();
      getModelsInfo();
    }
  }, [apiStatus]);
  
  return (
    <Router>
      <div className="app-container">
        <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
          <div className="container">
            <Link className="navbar-brand" to="/">
              Falsehood Filter
            </Link>
            <button
              className="navbar-toggler"
              type="button"
              data-bs-toggle="collapse"
              data-bs-target="#navbarNav"
              aria-controls="navbarNav"
              aria-expanded="false"
              aria-label="Toggle navigation"
            >
              <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarNav">
              <ul className="navbar-nav">
                <li className="nav-item">
                  <Link className="nav-link" to="/">
                    Home
                  </Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/upload-train">
                    Upload & Train
                  </Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/analyze">
                    Analyze Text
                  </Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/results">
                    Results
                  </Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/about">
                    About
                  </Link>
                </li>
              </ul>
            </div>
            
            {/* API Status Indicator */}
            <div className="api-status ms-auto">
              {apiStatus === 'checking' && (
                <span className="badge bg-secondary">Checking API...</span>
              )}
              {apiStatus === 'connected' && (
                <span className="badge bg-success">API Connected</span>
              )}
              {apiStatus === 'disconnected' && (
                <span className="badge bg-danger">API Disconnected</span>
              )}
            </div>
          </div>
        </nav>
        
        <main className="container my-4">
          {apiStatus === 'disconnected' && (
            <div className="alert alert-danger" role="alert">
              <h4 className="alert-heading">API Server Not Connected</h4>
              <p>
                Unable to connect to the API server. The application will not function properly.
                Please make sure the API server is running.
              </p>
            </div>
          )}
          
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route 
              path="/analyze" 
              element={<AnalyzeText trainedModels={trainedModels} />} 
            />
            <Route 
              path="/upload-train" 
              element={
                <UploadTrain 
                  dataset={dataset} 
                  setDataset={setDataset} 
                  setTrainedModels={setTrainedModels} 
                  setEvaluationResults={setEvaluationResults} 
                />
              } 
            />
            <Route 
              path="/results" 
              element={
                <ResultsDashboard 
                  dataset={dataset} 
                  trainedModels={trainedModels} 
                  evaluationResults={evaluationResults} 
                />
              } 
            />
          </Routes>
        </main>
        
        <footer className="bg-light p-3 text-center">
          <div className="container">
            <span className="text-muted">Falsehood Filter System © {new Date().getFullYear()}</span>
          </div>
        </footer>
        
        {/* Collaboration Panel */}
        <CollaborationPanel />
      </div>
    </Router>
  );
}

export default App;
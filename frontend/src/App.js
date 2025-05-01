import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';

// Import components for different pages
import Home from './components/Home';
import UploadTrain from './components/UploadTrain';
import AnalyzeText from './components/AnalyzeText';
import ResultsDashboard from './components/ResultsDashboard';
import About from './components/About';

function App() {
  const [dataset, setDataset] = useState(null);
  const [trainedModels, setTrainedModels] = useState([]);
  const [evaluationResults, setEvaluationResults] = useState({});
  
  return (
    <Router>
      <div className="App">
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
          <div className="container">
            <Link className="navbar-brand" to="/">Fake News Detection</Link>
            <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
              <span className="navbar-toggler-icon"></span>
            </button>
            <div className="collapse navbar-collapse" id="navbarNav">
              <ul className="navbar-nav">
                <li className="nav-item">
                  <Link className="nav-link" to="/">Home</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/upload-train">Upload & Train</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/analyze">Analyze Text</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/results">Results Dashboard</Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to="/about">About</Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>

        <div className="container mt-4">
          <Routes>
            <Route path="/" element={<Home />} />
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
              path="/analyze" 
              element={
                <AnalyzeText 
                  trainedModels={trainedModels} 
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
            <Route path="/about" element={<About />} />
          </Routes>
        </div>

        <footer className="footer mt-auto py-3 bg-dark text-white">
          <div className="container text-center">
            <span>Fake News Detection Application © {new Date().getFullYear()}</span>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
import React from 'react';
import { Link } from 'react-router-dom';
import TutorialButton from './TutorialButton';

const Home = () => {
  return (
    <div className="home-container">
      <TutorialButton pageName="home" />
      <div className="jumbotron bg-light p-5 rounded">
        <h1 className="display-4">Falsehood Filter System</h1>
        <p className="lead">
          A sophisticated machine learning application to detect false information using advanced NLP techniques.
        </p>
        <hr className="my-4" />
        <p>
          This application uses multiple state-of-the-art machine learning models to analyze and
          classify news content as authentic or false with high accuracy.
        </p>
        <div className="d-flex gap-2 mt-4">
          <Link to="/upload-train" className="btn btn-primary">
            Upload & Train
          </Link>
          <Link to="/analyze" className="btn btn-success">
            Analyze Text
          </Link>
        </div>
      </div>
      
      {/* Latest Improvements Section */}
      <div className="card mt-4 mb-4 border-primary">
        <div className="card-header bg-primary text-white">
          <h4 className="mb-0">Recent System Improvements</h4>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-md-6">
              <ul className="list-group list-group-flush">
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i> 
                  <strong>Comprehensive UI rebranding</strong> from "Fake News Detection" to "Falsehood Filter"
                </li>
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i> 
                  <strong>Implemented web scraping</strong> using Trafilatura to extract content from URLs
                </li>
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i> 
                  <strong>Enhanced preprocessing</strong> with robust error handling for missing NLTK resources
                </li>
              </ul>
            </div>
            <div className="col-md-6">
              <ul className="list-group list-group-flush">
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i> 
                  <strong>Added comparative analysis</strong> between advanced algorithms and traditional methods
                </li>
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i> 
                  <strong>Implemented metrics export</strong> for downloading performance data as CSV
                </li>
                <li className="list-group-item">
                  <i className="bi bi-check-circle-fill text-success me-2"></i>
                  <strong>Transitioned to React frontend</strong> with modern UI components and responsive design
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="row mt-3">
        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Advanced ML Models</h5>
              <p className="card-text">
                Our application leverages multiple advanced machine learning models:
              </p>
              <ul>
                <li>DeBERTa with Disentangled Attention</li>
                <li>Model-Agnostic Meta-Learning (MAML)</li>
                <li>Contrastive Learning (SimCLR, MoCo)</li>
                <li>Reinforcement Learning (DQN & Policy Gradient)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-4">
          <div className="card h-100">
            <div className="card-body">
              <h5 className="card-title">Key Features</h5>
              <p className="card-text">
                Take advantage of our powerful tools to detect false information:
              </p>
              <ul>
                <li>Upload and train models on your own dataset</li>
                <li>Analyze text with multiple ML models</li>
                <li>Compare model performance with detailed metrics</li>
                <li>Get confidence scores and predictions for news articles</li>
                <li>Web scraping capabilities for online content</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="row mt-3">
        <div className="col-md-4 mb-4">
          <div className="card text-center h-100">
            <div className="card-body">
              <h5 className="card-title">Upload & Train</h5>
              <p className="card-text">
                Upload your dataset and train our models to detect false information with high accuracy.
              </p>
              <Link to="/upload-train" className="btn btn-outline-primary">
                Get Started
              </Link>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-4">
          <div className="card text-center h-100">
            <div className="card-body">
              <h5 className="card-title">Analyze Text</h5>
              <p className="card-text">
                Input any news article to analyze and check if it contains authentic or false information.
              </p>
              <Link to="/analyze" className="btn btn-outline-success">
                Try It Now
              </Link>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-4">
          <div className="card text-center h-100">
            <div className="card-body">
              <h5 className="card-title">Results Dashboard</h5>
              <p className="card-text">
                View detailed performance metrics and compare different models.
              </p>
              <Link to="/results" className="btn btn-outline-info">
                Check Results
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
import React, { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const About = () => {
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchComparisonData = async () => {
      try {
        const response = await axios.get('/api/model-comparison');
        setComparisonData(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching model comparison data:', err);
        setError('Failed to load model comparison data');
        setLoading(false);
      }
    };

    fetchComparisonData();
  }, []);

  // Generate chart data from API response
  const getComparisonChartData = () => {
    if (!comparisonData) return null;

    return {
      labels: comparisonData.categories,
      datasets: [
        {
          label: 'Traditional Methods',
          data: comparisonData.traditional_scores,
          backgroundColor: 'rgba(255, 159, 64, 0.6)',
          borderColor: 'rgb(255, 159, 64)',
          borderWidth: 1
        },
        {
          label: 'Our Advanced Algorithms',
          data: comparisonData.advanced_scores,
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgb(54, 162, 235)',
          borderWidth: 1
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Advanced vs Traditional Methods Comparison',
        font: {
          size: 16
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Score (%)',
        },
      },
    },
  };

  return (
    <div className="about-container">
      <h2 className="page-title">About the Falsehood Filter System</h2>
      
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Project Overview</h3>
          <p>
            The Falsehood Filter System is a sophisticated tool that leverages state-of-the-art 
            machine learning and natural language processing techniques to analyze and classify news content.
            Our goal is to provide users with accurate and reliable tools to identify potentially misleading
            or false information in news articles.
          </p>
        </div>
      </div>
      
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Implemented Advanced Algorithms</h3>
          <p>
            This application uses multiple advanced machine learning models to achieve high accuracy in false information detection:
          </p>
          <div className="row">
            <div className="col-md-6">
              <h5>DeBERTa with Disentangled Attention</h5>
              <p>
                DeBERTa (Decoding-Enhanced BERT with Disentangled Attention) improves the BERT and RoBERTa 
                models using disentangled attention mechanisms and an enhanced mask decoder. It achieves 
                state-of-the-art results on multiple NLP tasks.
              </p>
            </div>
            <div className="col-md-6">
              <h5>Model-Agnostic Meta-Learning (MAML)</h5>
              <p>
                MAML is designed to quickly adapt to new tasks with minimal training data. This meta-learning
                approach allows the model to learn how to learn, making it efficient for false information detection
                across different domains and sources.
              </p>
            </div>
          </div>
          <div className="row mt-3">
            <div className="col-md-6">
              <h5>Contrastive Learning</h5>
              <p>
                Inspired by SimCLR and MoCo, our contrastive learning approach creates robust representations
                by pulling together similar examples (same class) and pushing apart dissimilar ones (different classes).
                This helps the model distinguish between subtle differences in real and false information.
              </p>
            </div>
            <div className="col-md-6">
              <h5>Reinforcement Learning</h5>
              <p>
                Using techniques like Deep Q-Networks (DQN) and Policy Gradient methods, our RL model learns
                optimal strategies for false information detection through trial and error, continuously improving
                as it processes more examples.
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Advanced vs Traditional Methods Comparison Section */}
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Advanced vs Traditional Methods</h3>
          
          {loading ? (
            <div className="d-flex justify-content-center">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
            </div>
          ) : error ? (
            <div className="alert alert-danger">{error}</div>
          ) : comparisonData ? (
            <>
              <p className="mb-4">
                Our advanced algorithms significantly outperform traditional false information detection methods
                across multiple key performance metrics:
              </p>
              
              {/* Performance Comparison Chart */}
              <div className="chart-container mb-4" style={{ height: '350px' }}>
                <Bar data={getComparisonChartData()} options={chartOptions} />
              </div>
              
              {/* Side by Side Comparison */}
              <div className="row mb-4">
                <div className="col-md-6">
                  <div className="card h-100 bg-light">
                    <div className="card-header bg-warning text-dark">
                      <h5 className="mb-0">Traditional Methods</h5>
                    </div>
                    <div className="card-body">
                      <ul className="list-group list-group-flush">
                        {comparisonData.traditional_models.map((model, index) => (
                          <li key={index} className="list-group-item bg-transparent">{model}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card h-100 bg-light">
                    <div className="card-header bg-primary text-white">
                      <h5 className="mb-0">Our Advanced Algorithms</h5>
                    </div>
                    <div className="card-body">
                      <ul className="list-group list-group-flush">
                        {comparisonData.advanced_models.map((model, index) => (
                          <li key={index} className="list-group-item bg-transparent">{model}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Key Advantages Section */}
              <h4 className="mb-3">Why Our Advanced Algorithms Are Better</h4>
              <div className="row">
                {comparisonData.advantages.map((advantage, index) => (
                  <div key={index} className="col-md-6 mb-3">
                    <div className="card h-100 border-primary">
                      <div className="card-header bg-primary text-white">
                        {advantage.title}
                      </div>
                      <div className="card-body">
                        <p className="card-text">{advantage.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Detailed Metrics Table */}
              <h4 className="mt-4 mb-3">Detailed Metric Comparison</h4>
              <div className="table-responsive">
                <table className="table table-bordered table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Metric</th>
                      <th>Traditional Methods</th>
                      <th>Our Advanced Algorithms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(comparisonData.detailed_metrics).map(([key, value]) => (
                      <tr key={key}>
                        <td>
                          <strong>{value.description}</strong>
                        </td>
                        <td>{value.traditional}</td>
                        <td className="table-success">{value.advanced}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : null}
        </div>
      </div>
      
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Technical Implementation</h3>
          <p>
            The application is built with a modern tech stack:
          </p>
          <ul>
            <li><strong>Frontend:</strong> React with Bootstrap for responsive design</li>
            <li><strong>Backend:</strong> Flask API serving ML models</li>
            <li><strong>ML Models:</strong> Custom implementations with fallback mechanisms for environments without specialized libraries</li>
            <li><strong>Data Processing:</strong> NLTK and custom preprocessing pipelines</li>
            <li><strong>Visualization:</strong> Chart.js for metrics visualization</li>
          </ul>
        </div>
      </div>
      
      <div className="card">
        <div className="card-body">
          <h3 className="card-title">How to Use</h3>
          <ol>
            <li>
              <strong>Upload & Train:</strong> Start by uploading a dataset of news articles labeled as real or false.
              Train one or more models on this dataset.
            </li>
            <li>
              <strong>Analyze Text:</strong> Paste any news article text or URL to analyze and get predictions
              on whether it contains real or false information, along with confidence scores.
            </li>
            <li>
              <strong>View Results:</strong> Check the Results Dashboard to compare model performance metrics
              and understand how each model performs.
            </li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default About;
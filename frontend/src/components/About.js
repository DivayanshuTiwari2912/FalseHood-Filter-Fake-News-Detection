import React from 'react';

const About = () => {
  return (
    <div className="about-container">
      <h2 className="page-title">About Fake News Detection</h2>
      
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Project Overview</h3>
          <p>
            The Fake News Detection application is a sophisticated tool that leverages state-of-the-art 
            machine learning and natural language processing techniques to analyze and classify news content.
            Our goal is to provide users with accurate and reliable tools to identify potentially misleading
            or false information in news articles.
          </p>
        </div>
      </div>
      
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">Implemented Algorithms</h3>
          <p>
            This application uses multiple advanced machine learning models to achieve high accuracy in fake news detection:
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
                approach allows the model to learn how to learn, making it efficient for fake news detection
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
                This helps the model distinguish between subtle differences in real and fake news.
              </p>
            </div>
            <div className="col-md-6">
              <h5>Reinforcement Learning</h5>
              <p>
                Using techniques like Deep Q-Networks (DQN) and Policy Gradient methods, our RL model learns
                optimal strategies for fake news detection through trial and error, continuously improving
                as it processes more examples.
              </p>
            </div>
          </div>
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
              <strong>Upload & Train:</strong> Start by uploading a dataset of news articles labeled as real or fake.
              Train one or more models on this dataset.
            </li>
            <li>
              <strong>Analyze Text:</strong> Paste any news article text or URL to analyze and get predictions
              on whether it's real or fake news, along with confidence scores.
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
import React, { useState, useEffect } from 'react';
import { FiX, FiArrowRight, FiArrowLeft, FiHelpCircle } from 'react-icons/fi';

// Interactive tutorial modal component
const TutorialModal = ({ isOpen, onClose, currentPage }) => {
  const [step, setStep] = useState(0);
  const [tutorialSteps, setTutorialSteps] = useState([]);

  // Initialize the tutorial steps based on current page
  useEffect(() => {
    if (!currentPage) return;
    
    // Define tutorial steps for each page
    const tutorialContentMap = {
      'home': [
        {
          title: 'Welcome to Falsehood Filter',
          content: 'This tool helps you analyze content and determine if it contains authentic or false information using advanced machine learning algorithms.',
          image: '/tutorial/home-welcome.png'
        },
        {
          title: 'Main Features',
          content: 'From this home page, you can access all the key features including text analysis, dataset training, and results dashboard.',
          image: '/tutorial/home-features.png'
        },
        {
          title: 'Getting Started',
          content: 'To begin analyzing text, click on the "Try It Now" button under the Analyze Text section.',
          image: '/tutorial/home-getting-started.png'
        }
      ],
      'analyze': [
        {
          title: 'Analyze Text',
          content: 'Here you can analyze news articles or any content to check if it contains false information.',
          image: '/tutorial/analyze-intro.png'
        },
        {
          title: 'Input Methods',
          content: 'You can either manually paste text or use the web scraper to analyze content directly from a website.',
          image: '/tutorial/analyze-input.png'
        },
        {
          title: 'Model Selection',
          content: 'Choose from different AI models. Each has different strengths in detecting various types of false information.',
          image: '/tutorial/analyze-model.png'
        },
        {
          title: 'Results and Sharing',
          content: 'After analysis, you\'ll see the result with confidence score and emoji rating. You can share these results via social media or email.',
          image: '/tutorial/analyze-results.png'
        }
      ],
      'upload-train': [
        {
          title: 'Upload & Train',
          content: 'This page allows you to upload your own dataset and train AI models to detect false information.',
          image: '/tutorial/upload-intro.png'
        },
        {
          title: 'Dataset Upload',
          content: 'Upload a CSV file containing examples of authentic and false information. You can drag and drop files or click to browse.',
          image: '/tutorial/upload-dataset.png'
        },
        {
          title: 'Model Training',
          content: 'Select a model to train, adjust parameters like training epochs, and click the Train button.',
          image: '/tutorial/upload-train.png'
        }
      ],
      'results': [
        {
          title: 'Results Dashboard',
          content: 'This dashboard shows the performance metrics of the trained models.',
          image: '/tutorial/results-intro.png'
        },
        {
          title: 'Model Comparison',
          content: 'Compare performance of different models across metrics like accuracy, precision, and recall.',
          image: '/tutorial/results-comparison.png'
        },
        {
          title: 'Export Results',
          content: 'You can download the performance metrics for further analysis or reporting.',
          image: '/tutorial/results-export.png'
        }
      ]
    };
    
    // Set tutorial steps based on current page
    setTutorialSteps(tutorialContentMap[currentPage] || []);
    
    // Reset step when changing pages
    setStep(0);
  }, [currentPage]);
  
  if (!isOpen || tutorialSteps.length === 0) return null;
  
  const currentStep = tutorialSteps[step];
  
  return (
    <div className="tutorial-modal-overlay">
      <div className="tutorial-modal">
        <button className="tutorial-close-btn" onClick={onClose}>
          <FiX />
        </button>
        
        <div className="tutorial-content">
          <h3>{currentStep.title}</h3>
          <p>{currentStep.content}</p>
          
          {/* Placeholder for images - these would be actual screenshots in a full implementation */}
          <div className="tutorial-image-placeholder">
            <FiHelpCircle size={40} />
            <p className="text-muted text-center">
              {currentStep.image ? currentStep.image.split('/').pop().replace('.png', '') : 'Tutorial visualization'}
            </p>
          </div>
          
          <div className="tutorial-navigation">
            <button 
              className="btn btn-outline-secondary" 
              onClick={() => setStep(prev => Math.max(0, prev - 1))}
              disabled={step === 0}
            >
              <FiArrowLeft /> Previous
            </button>
            
            <div className="tutorial-progress">
              {tutorialSteps.map((_, index) => (
                <span 
                  key={index} 
                  className={`tutorial-dot ${index === step ? 'active' : ''}`}
                  onClick={() => setStep(index)}
                />
              ))}
            </div>
            
            {step < tutorialSteps.length - 1 ? (
              <button 
                className="btn btn-primary" 
                onClick={() => setStep(prev => Math.min(tutorialSteps.length - 1, prev + 1))}
              >
                Next <FiArrowRight />
              </button>
            ) : (
              <button className="btn btn-success" onClick={onClose}>
                Finish
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TutorialModal;
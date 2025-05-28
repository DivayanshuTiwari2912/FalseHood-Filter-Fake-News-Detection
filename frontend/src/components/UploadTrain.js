import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiFile, FiCheckCircle } from 'react-icons/fi';
import apiService from '../services/api';
import TutorialButton from './TutorialButton';

const UploadTrain = ({ dataset, setDataset, setTrainedModels, setEvaluationResults }) => {
  const [file, setFile] = useState(null);
  const [textColumn, setTextColumn] = useState('text');
  const [labelColumn, setLabelColumn] = useState('label');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  
  const [selectedModels, setSelectedModels] = useState([]);
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(32);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingError, setTrainingError] = useState('');
  const [trainingSuccess, setTrainingSuccess] = useState('');
  
  const [availableModels, setAvailableModels] = useState({});
  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState('');
  
  const fileInputRef = useRef(null);
  
  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await apiService.getAvailableModels();
        setAvailableModels(response.data.available_models);
        setTrainedModels(response.data.trained_models || []);
      } catch (err) {
        console.error('Error fetching models:', err);
      }
    };
    
    fetchModels();
  }, [setTrainedModels]);
  
  // Handle dropped files from react-dropzone
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setUploadError('');
    }
  }, []);

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.csv']
    },
    maxFiles: 1
  });

  // Traditional file input handler (as backup)
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadError('');
    }
  };
  
  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setUploadError('Please select a file to upload.');
      return;
    }
    
    setIsUploading(true);
    setUploadError('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('text_col', textColumn);
      formData.append('label_col', labelColumn);
      
      const response = await apiService.uploadDataset(formData);
      setDataset(response.data.dataset);
      setIsUploading(false);
    } catch (err) {
      setIsUploading(false);
      console.error('Error uploading dataset:', err);
      
      if (err.response && err.response.data && err.response.data.error) {
        setUploadError(err.response.data.error);
      } else {
        setUploadError('Error uploading dataset. Please try again.');
      }
    }
  };
  
  const handleTrain = async (e) => {
    e.preventDefault();
    
    if (!dataset) {
      setTrainingError('Please upload a dataset first.');
      return;
    }
    if (!selectedModels.length) {
      setTrainingError('Please select at least one model to train.');
      return;
    }

    setIsTraining(true);
    setTrainingError('');
    setTrainingSuccess('');

    let successMessages = [];
    let errorMessages = [];

    for (const model of selectedModels) {
      try {
        const response = await apiService.trainModel(model, epochs, batchSize);
        successMessages.push(`Model ${model} trained successfully!`);
        // Update trained models list after each
        const modelsResponse = await apiService.getAvailableModels();
        setTrainedModels(modelsResponse.data.trained_models || []);
        // Automatically evaluate the model
        await handleEvaluate(model);
      } catch (err) {
        if (err.response && err.response.data && err.response.data.error) {
          errorMessages.push(err.response.data.error);
        } else {
          errorMessages.push(`Error training model ${model}. Please try again.`);
        }
      }
    }

    setIsTraining(false);
    setTrainingSuccess(successMessages.join(' '));
    setTrainingError(errorMessages.join(' '));
  };
  
  const handleEvaluate = async (modelName) => {
    setEvaluating(true);
    setEvaluationError('');
    
    try {
      const response = await apiService.evaluateModel(modelName);
      
      // Update evaluation results in parent component
      setEvaluationResults(prev => ({
        ...prev,
        [modelName]: response.data.metrics
      }));
      
      setEvaluating(false);
    } catch (err) {
      setEvaluating(false);
      console.error('Error evaluating model:', err);
      
      if (err.response && err.response.data && err.response.data.error) {
        setEvaluationError(err.response.data.error);
      } else {
        setEvaluationError('Error evaluating model. Please try again.');
      }
    }
  };
  
  return (
    <div className="upload-train-container">
      <h2 className="page-title">Upload Dataset & Train Models</h2>
      <TutorialButton pageName="upload-train" />
      
      {/* Upload Dataset Section */}
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">1. Upload Dataset</h3>
          <p className="card-text">
            Upload a CSV file containing news articles with labels indicating whether each article is real or fake.
          </p>
          
          <form onSubmit={handleUpload}>
            <div className="mb-3">
              <label className="form-label">Dataset File (CSV)</label>
              <div 
                {...getRootProps()} 
                className={`dropzone border rounded p-4 text-center ${isDragActive ? 'border-primary bg-light' : 'border-dashed'}`}
                style={{
                  borderStyle: 'dashed',
                  cursor: 'pointer',
                  transition: 'border .3s ease-in-out, background-color .3s ease-in-out'
                }}
              >
                <input {...getInputProps()} />
                
                {isDragActive ? (
                  <div className="py-4">
                    <FiUpload className="h3 mb-2 text-primary" />
                    <p>Drop the CSV file here...</p>
                  </div>
                ) : file ? (
                  <div className="py-2">
                    <FiCheckCircle className="h3 mb-2 text-success" />
                    <p className="mb-1">File selected:</p>
                    <p className="text-primary font-weight-bold">{file.name}</p>
                  </div>
                ) : (
                  <div className="py-4">
                    <FiUpload className="h3 mb-2" />
                    <p className="mb-1">Drag & drop a CSV file here, or click to select</p>
                    <p className="small text-muted mb-0">Supported format: CSV</p>
                  </div>
                )}
              </div>
              <div className="form-text mt-2">
                Please ensure your CSV file has columns for text content and labels (0 for fake, 1 for real).
              </div>
            </div>
            
            <div className="row">
              <div className="col-md-6">
                <div className="mb-3">
                  <label htmlFor="text-column" className="form-label">
                    Text Column Name
                  </label>
                  <input
                    type="text"
                    className="form-control"
                    id="text-column"
                    value={textColumn}
                    onChange={(e) => setTextColumn(e.target.value)}
                    placeholder="Default: text"
                  />
                </div>
              </div>
              
              <div className="col-md-6">
                <div className="mb-3">
                  <label htmlFor="label-column" className="form-label">
                    Label Column Name
                  </label>
                  <input
                    type="text"
                    className="form-control"
                    id="label-column"
                    value={labelColumn}
                    onChange={(e) => setLabelColumn(e.target.value)}
                    placeholder="Default: label"
                  />
                </div>
              </div>
            </div>
            
            {uploadError && (
              <div className="alert alert-danger" role="alert">
                {uploadError}
              </div>
            )}
            
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isUploading || !file}
            >
              {isUploading ? (
                <>
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                  &nbsp;Uploading...
                </>
              ) : (
                'Upload Dataset'
              )}
            </button>
          </form>
          
          {dataset && (
            <div className="alert alert-success mt-3">
              <h5>Dataset Uploaded Successfully!</h5>
              <p>
                <strong>Filename:</strong> {dataset.filename}
                <br />
                <strong>Total Samples:</strong> {dataset.num_samples}
                <br />
                <strong>Training Samples:</strong> {dataset.num_train}
                <br />
                <strong>Testing Samples:</strong> {dataset.num_test}
              </p>
            </div>
          )}
        </div>
      </div>
      
      {/* Train Model Section */}
      <div className="card">
        <div className="card-body">
          <h3 className="card-title">2. Train Model</h3>
          <p className="card-text">
            Select a model to train on the uploaded dataset.
          </p>
          
          {!dataset && (
            <div className="alert alert-warning">
              Please upload a dataset first before training a model.
            </div>
          )}
          
          <form onSubmit={handleTrain}>
            <div className="mb-3">
              <label htmlFor="model-select" className="form-label">
                Select Models
              </label>
              <select
                id="model-select"
                className="form-select"
                multiple
                value={selectedModels}
                onChange={(e) => {
                  const options = Array.from(e.target.selectedOptions, option => option.value);
                  setSelectedModels(options);
                }}
              >
                {Object.entries(availableModels).map(([key, name]) => (
                  <option key={key} value={key}>
                    {name}
                  </option>
                ))}
              </select>
              <div className="form-text">Hold Ctrl (Windows) or Cmd (Mac) to select multiple models.</div>
            </div>
            
            <div className="row">
              <div className="col-md-6">
                <div className="mb-3">
                  <label htmlFor="epochs" className="form-label">
                    Training Epochs
                  </label>
                  <input
                    type="number"
                    className="form-control"
                    id="epochs"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    min="1"
                    max="20"
                  />
                </div>
              </div>
              
              <div className="col-md-6">
                <div className="mb-3">
                  <label htmlFor="batch-size" className="form-label">
                    Batch Size
                  </label>
                  <input
                    type="number"
                    className="form-control"
                    id="batch-size"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    min="1"
                    max="128"
                  />
                </div>
              </div>
            </div>
            
            {trainingError && (
              <div className="alert alert-danger" role="alert">
                {trainingError}
              </div>
            )}
            
            {trainingSuccess && (
              <div className="alert alert-success" role="alert">
                {trainingSuccess}
              </div>
            )}
            
            {evaluationError && (
              <div className="alert alert-warning" role="alert">
                {evaluationError}
              </div>
            )}
            
            <button
              type="submit"
              className="btn btn-success"
              disabled={isTraining || !dataset}
            >
              {isTraining ? (
                <>
                  <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                  &nbsp;Training...
                </>
              ) : (
                'Train Model'
              )}
            </button>
            
            {evaluating && (
              <div className="mt-3">
                <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                &nbsp;Evaluating model performance...
              </div>
            )}
          </form>
          
          <div className="mt-4">
            <Link to="/analyze" className="btn btn-outline-primary">
              Go to Analyze Text
            </Link>
            <Link to="/results" className="btn btn-outline-info ms-2">
              View Results Dashboard
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadTrain;
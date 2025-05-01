import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Bar, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

const ResultsDashboard = ({ dataset, trainedModels, evaluationResults }) => {
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  
  // Colors for different models
  const modelColors = {
    deberta: { bg: 'rgba(54, 162, 235, 0.2)', border: 'rgb(54, 162, 235)' },
    maml: { bg: 'rgba(255, 99, 132, 0.2)', border: 'rgb(255, 99, 132)' },
    contrastive: { bg: 'rgba(75, 192, 192, 0.2)', border: 'rgb(75, 192, 192)' },
    rl: { bg: 'rgba(255, 159, 64, 0.2)', border: 'rgb(255, 159, 64)' },
  };
  
  // Generate data for Bar Chart
  const getBarChartData = () => {
    const labels = trainedModels;
    const data = {
      labels,
      datasets: [
        {
          label: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
          data: labels.map(model => 
            evaluationResults[model] ? evaluationResults[model][selectedMetric] * 100 : 0
          ),
          backgroundColor: labels.map(model => modelColors[model]?.bg || 'rgba(153, 102, 255, 0.2)'),
          borderColor: labels.map(model => modelColors[model]?.border || 'rgb(153, 102, 255)'),
          borderWidth: 1,
        },
      ],
    };
    return data;
  };
  
  // Generate data for Radar Chart
  const getRadarChartData = () => {
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const data = {
      labels: metrics.map(metric => metric.charAt(0).toUpperCase() + metric.slice(1)),
      datasets: trainedModels.map(model => ({
        label: model.charAt(0).toUpperCase() + model.slice(1),
        data: metrics.map(metric => 
          evaluationResults[model] ? evaluationResults[model][metric] * 100 : 0
        ),
        backgroundColor: modelColors[model]?.bg || 'rgba(153, 102, 255, 0.2)',
        borderColor: modelColors[model]?.border || 'rgb(153, 102, 255)',
        borderWidth: 1,
      })),
    };
    return data;
  };
  
  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
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
  
  const radarOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Model Metrics Comparison',
      },
    },
    scales: {
      r: {
        angleLines: {
          display: true,
        },
        suggestedMin: 0,
        suggestedMax: 100,
      },
    },
  };
  
  return (
    <div className="results-dashboard-container">
      <h2 className="page-title">Results Dashboard</h2>
      
      {trainedModels.length === 0 ? (
        <div className="alert alert-info">
          <p>No trained models available. Please train some models first to see results.</p>
          <Link to="/upload-train" className="btn btn-primary">
            Go to Upload & Train
          </Link>
        </div>
      ) : (
        <>
          {dataset && (
            <div className="card mb-4">
              <div className="card-body">
                <h3 className="card-title">Dataset Information</h3>
                <div className="row">
                  <div className="col-md-6">
                    <p><strong>Filename:</strong> {dataset.filename}</p>
                    <p><strong>Total Samples:</strong> {dataset.num_samples}</p>
                  </div>
                  <div className="col-md-6">
                    <p><strong>Training Set:</strong> {dataset.num_train} samples</p>
                    <p><strong>Test Set:</strong> {dataset.num_test} samples</p>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="card mb-4">
            <div className="card-body">
              <h3 className="card-title">Model Performance</h3>
              
              <div className="mb-3">
                <label htmlFor="metric-select" className="form-label">
                  Select Metric for Comparison
                </label>
                <select
                  id="metric-select"
                  className="form-select"
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                  <option value="f1_score">F1 Score</option>
                </select>
              </div>
              
              <div className="chart-container" style={{ height: '300px' }}>
                <Bar options={barOptions} data={getBarChartData()} />
              </div>
            </div>
          </div>
          
          <div className="card mb-4">
            <div className="card-body">
              <h3 className="card-title">Model Metrics Comparison</h3>
              <div className="chart-container" style={{ height: '300px' }}>
                <Radar options={radarOptions} data={getRadarChartData()} />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="card-body">
              <h3 className="card-title">Detailed Metrics</h3>
              <div className="table-responsive">
                <table className="table table-bordered">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1 Score</th>
                      <th>Samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainedModels.map(model => (
                      <tr key={model}>
                        <td>{model.charAt(0).toUpperCase() + model.slice(1)}</td>
                        <td>{evaluationResults[model] ? (evaluationResults[model].accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td>{evaluationResults[model] ? (evaluationResults[model].precision * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td>{evaluationResults[model] ? (evaluationResults[model].recall * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td>{evaluationResults[model] ? (evaluationResults[model].f1_score * 100).toFixed(2) + '%' : 'N/A'}</td>
                        <td>{evaluationResults[model] ? evaluationResults[model].num_samples : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          
          <div className="mt-4">
            <Link to="/analyze" className="btn btn-primary">
              Analyze New Text
            </Link>
          </div>
        </>
      )}
    </div>
  );
};

export default ResultsDashboard;
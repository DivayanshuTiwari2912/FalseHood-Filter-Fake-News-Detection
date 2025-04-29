import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict method
        X_test (array-like): Test features
        y_test (array-like): Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred, confidences = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary')
    }
    
    return metrics

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple trained models on test data.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (array-like): Test features
        y_test (array-like): Test labels
        
    Returns:
        dict: Dictionary mapping model names to evaluation metrics
    """
    results = {}
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        
        print(f"Metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results

def evaluate_ensemble(models, X_test, y_test, method='majority'):
    """
    Evaluate an ensemble of models using voting.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (array-like): Test features
        y_test (array-like): Test labels
        method (str): Voting method ('majority' or 'weighted')
        
    Returns:
        dict: Dictionary of evaluation metrics for the ensemble
    """
    # Get predictions from each model
    predictions = {}
    confidences = {}
    
    for model_name, model in models.items():
        y_pred, confs = model.predict(X_test)
        predictions[model_name] = y_pred
        confidences[model_name] = confs
    
    # Combine predictions
    if method == 'majority':
        # Simple majority voting
        ensemble_pred = np.zeros(len(y_test))
        
        for model_preds in predictions.values():
            ensemble_pred += model_preds
        
        ensemble_pred = (ensemble_pred >= len(models) / 2).astype(int)
        
    elif method == 'weighted':
        # Weighted voting based on model confidence
        ensemble_pred = np.zeros(len(y_test))
        
        for model_name in models:
            model_preds = predictions[model_name]
            model_confs = confidences[model_name]
            
            # Add weighted votes
            for i in range(len(model_preds)):
                if model_preds[i] == 1:
                    ensemble_pred[i] += model_confs[i]
                else:
                    ensemble_pred[i] -= model_confs[i]
        
        ensemble_pred = (ensemble_pred > 0).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'precision': precision_score(y_test, ensemble_pred, average='binary'),
        'recall': recall_score(y_test, ensemble_pred, average='binary'),
        'f1': f1_score(y_test, ensemble_pred, average='binary')
    }
    
    print(f"Metrics for Ensemble ({method} voting):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics

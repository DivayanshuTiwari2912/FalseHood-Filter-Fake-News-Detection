import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, X_train, y_train, epochs=3, batch_size=32, verbose=True):
    """
    Train a model on the given data.
    
    Args:
        model: Model to train
        X_train (array-like): Training texts
        y_train (array-like): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        verbose (bool): Whether to print progress
        
    Returns:
        object: Trained model
    """
    if verbose:
        print(f"Training {model.__class__.__name__}...")
        print(f"Data size: {len(X_train)} samples")
        print(f"Training parameters: epochs={epochs}, batch_size={batch_size}")
        
    start_time = time.time()
    
    # Train the model
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    end_time = time.time()
    
    if verbose:
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return model

def train_models(models, X_train, y_train, epochs=3, batch_size=32, verbose=True):
    """
    Train multiple models on the same data.
    
    Args:
        models (list): List of models to train
        X_train (array-like): Training texts
        y_train (array-like): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Dictionary of trained models
    """
    trained_models = {}
    
    for model in models:
        model_name = model.__class__.__name__
        if verbose:
            print(f"\nTraining {model_name}...")
        
        trained_model = train_model(
            model, X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose
        )
        
        trained_models[model_name] = trained_model
    
    return trained_models

def train_and_evaluate(models, X_train, y_train, X_test, y_test, epochs=3, batch_size=32):
    """
    Train and evaluate multiple models.
    
    Args:
        models (list): List of models to train
        X_train (array-like): Training texts
        y_train (array-like): Training labels
        X_test (array-like): Testing texts
        y_test (array-like): Testing labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        dict: Dictionary mapping model names to (model, metrics) tuples
    """
    results = {}
    
    for model in models:
        model_name = model.__class__.__name__
        print(f"\n=== Training {model_name} ===")
        
        # Train the model
        trained_model = train_model(
            model, X_train, y_train,
            epochs=epochs, batch_size=batch_size
        )
        
        # Evaluate the model
        y_pred, _ = trained_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary')
        }
        
        print(f"Metrics for {model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        results[model_name] = (trained_model, metrics)
    
    return results

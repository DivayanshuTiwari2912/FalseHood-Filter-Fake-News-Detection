import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_performance_metrics(models_metrics):
    """
    Plot performance metrics for multiple models.
    
    Args:
        models_metrics (dict): Dictionary mapping model names to metric dictionaries
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract model names and metrics
    model_names = list(models_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Plot accuracy
    accuracy_values = [models_metrics[model]['accuracy'] for model in model_names]
    axs[0].bar(model_names, accuracy_values, color='royalblue')
    axs[0].set_title('Accuracy')
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Score')
    
    # Plot precision and recall
    precision_values = [models_metrics[model]['precision'] for model in model_names]
    recall_values = [models_metrics[model]['recall'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axs[1].bar(x - width/2, precision_values, width, label='Precision', color='green')
    axs[1].bar(x + width/2, recall_values, width, label='Recall', color='orange')
    axs[1].set_title('Precision & Recall')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(model_names)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Score')
    axs[1].legend()
    
    # Plot F1 Score
    f1_values = [models_metrics[model]['f1'] for model in model_names]
    axs[2].bar(model_names, f1_values, color='purple')
    axs[2].set_title('F1 Score')
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Score')
    
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=['Fake', 'Real']):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): Names of the classes
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    return fig

def plot_confidence_distribution(model, X_test, y_test):
    """
    Plot the distribution of prediction confidences.
    
    Args:
        model: The trained model
        X_test (array-like): Test features
        y_test (array-like): Test labels
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get predictions and confidences
    y_pred, confidences = model.predict(X_test)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Separate confidences by correct and incorrect predictions
    correct = confidences[y_pred == y_test]
    incorrect = confidences[y_pred != y_test]
    
    # Plot histograms
    ax.hist(correct, bins=10, alpha=0.5, label='Correct Predictions', color='green')
    ax.hist(incorrect, bins=10, alpha=0.5, label='Incorrect Predictions', color='red')
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    
    return fig

def plot_roc_curve(models_data):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_data (dict): Dictionary mapping model names to (y_true, y_scores) tuples
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, (y_true, y_scores) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    return fig

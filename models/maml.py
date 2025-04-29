import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import torch, but provide fallbacks if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using simplified implementation for MAML.")

# Define PyTorch modules only if torch is available
if TORCH_AVAILABLE:
    class TaskEncoder(nn.Module):
        """Simple encoder for text classification tasks in MAML."""
        
        def __init__(self, input_dim, hidden_dim=64):
            super(TaskEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )
            
        def forward(self, x):
            return self.encoder(x)

    class TaskClassifier(nn.Module):
        """Task-specific classifier for MAML."""
        
        def __init__(self, input_dim, n_classes=2):
            super(TaskClassifier, self).__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, n_classes)
            )
            
        def forward(self, x):
            return self.classifier(x)
else:
    # Dummy classes when torch is not available
    class TaskEncoder:
        pass
        
    class TaskClassifier:
        pass

class MAMLModel:
    """
    Implementation of Model-Agnostic Meta-Learning (MAML) for fake news detection.
    
    This is a simplified version of MAML, adapted for text classification.
    """
    
    def __init__(self, max_features=5000, hidden_dim=64):
        """
        Initialize the MAML model.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            hidden_dim (int): Dimension of hidden layers
        """
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Check if torch is available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Meta-parameters
            self.meta_lr = 0.01
            self.task_lr = 0.1
            self.meta_batch_size = 5
            
            # Initialize models
            self.encoder = None
            self.classifier = None
            self.is_simplified = False
        else:
            # Use simplified implementation
            self.is_simplified = True
            self.word_weights = {}
    
    def train(self, X_train, y_train, epochs=3, batch_size=32):
        """
        Train the MAML model.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            self: Trained model
        """
        # Check if using simplified implementation
        if self.is_simplified:
            self._train_simplified(X_train, y_train)
            return self
        
        # Fit and transform training data
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        input_dim = X_train_vec.shape[1]
        
        # Initialize models if not already done
        if self.encoder is None:
            self.encoder = TaskEncoder(input_dim, self.hidden_dim).to(self.device)
        
        if self.classifier is None:
            self.classifier = TaskClassifier(self.hidden_dim // 2).to(self.device)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train_vec).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizers
        meta_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.meta_lr
        )
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                # Move tensors to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Meta-learning: simulate multiple tasks by splitting the batch
                meta_batch_loss = 0
                
                for i in range(0, len(batch_X), self.meta_batch_size):
                    # Create a "task" from a subset of the batch
                    task_X = batch_X[i:i + self.meta_batch_size]
                    task_y = batch_y[i:i + self.meta_batch_size]
                    
                    if len(task_X) < 2:  # Skip if not enough samples
                        continue
                    
                    # Split into support and query sets
                    split_idx = len(task_X) // 2
                    support_X, query_X = task_X[:split_idx], task_X[split_idx:]
                    support_y, query_y = task_y[:split_idx], task_y[split_idx:]
                    
                    # Clone model parameters for task-specific adaptation
                    task_encoder = TaskEncoder(input_dim, self.hidden_dim).to(self.device)
                    task_classifier = TaskClassifier(self.hidden_dim // 2).to(self.device)
                    
                    # Copy parameters
                    task_encoder.load_state_dict(self.encoder.state_dict())
                    task_classifier.load_state_dict(self.classifier.state_dict())
                    
                    # Task-specific adaptation
                    # Forward pass on support set
                    support_features = task_encoder(support_X)
                    support_logits = task_classifier(support_features)
                    support_loss = F.cross_entropy(support_logits, support_y)
                    
                    # Update task-specific parameters
                    grads = torch.autograd.grad(support_loss, 
                                              list(task_encoder.parameters()) + list(task_classifier.parameters()),
                                              create_graph=True)
                    
                    # Manual parameter update
                    task_params = list(task_encoder.parameters()) + list(task_classifier.parameters())
                    for param, grad in zip(task_params, grads):
                        param.data = param.data - self.task_lr * grad
                    
                    # Evaluate on query set
                    query_features = task_encoder(query_X)
                    query_logits = task_classifier(query_features)
                    query_loss = F.cross_entropy(query_logits, query_y)
                    
                    meta_batch_loss += query_loss
                
                # Meta-update
                if meta_batch_loss > 0:  # Only update if we had valid tasks
                    meta_optimizer.zero_grad()
                    meta_batch_loss.backward()
                    meta_optimizer.step()
                    
                    total_loss += meta_batch_loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def _train_simplified(self, X_train, y_train):
        """
        Simplified training method for demonstration when PyTorch is not available.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
        """
        # Use TF-IDF to extract features
        self.vectorizer.fit(X_train)
        
        # Count word frequencies for each class
        word_counts = {0: {}, 1: {}}
        class_counts = {0: 0, 1: 0}
        
        for text, label in zip(X_train, y_train):
            class_counts[label] += 1
            
            # Get TF-IDF features
            features = self.vectorizer.transform([text]).toarray()[0]
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Update word weights
            for idx, value in enumerate(features):
                if value > 0:
                    word = feature_names[idx]
                    if word not in word_counts[label]:
                        word_counts[label][word] = 0
                    word_counts[label][word] += value
        
        # Calculate word weights
        self.word_weights = {}
        vocab = set()
        for label_words in word_counts.values():
            vocab.update(label_words.keys())
        
        for word in vocab:
            # Add smoothing to avoid division by zero
            fake_weight = (word_counts[0].get(word, 0) + 0.1)
            real_weight = (word_counts[1].get(word, 0) + 0.1)
            
            # Log ratio as a simple feature weight
            self.word_weights[word] = np.log((real_weight / class_counts[1]) / 
                                            (fake_weight / class_counts[0]))
        
        # Save class priors
        total_docs = class_counts[0] + class_counts[1]
        self.class_prior = {
            0: class_counts[0] / total_docs,
            1: class_counts[1] / total_docs
        }
    
    def predict(self, X):
        """
        Predict fake/real labels for input texts.
        
        Args:
            X (array-like): Input texts
            
        Returns:
            tuple: (predictions, confidences)
        """
        # Check if using simplified implementation
        if self.is_simplified:
            return self._predict_simplified(X)
        
        # Transform input data
        X_vec = self.vectorizer.transform(X).toarray()
        X_tensor = torch.FloatTensor(X_vec).to(self.device)
        
        # Set models to evaluation mode
        self.encoder.eval()
        self.classifier.eval()
        
        # Make predictions
        with torch.no_grad():
            features = self.encoder(X_tensor)
            logits = self.classifier(features)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions and confidences
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = probs.max(dim=1)[0].cpu().numpy()
        
        return preds, confs
    
    def _predict_simplified(self, X):
        """
        Simplified prediction method for demonstration.
        
        Args:
            X (array-like): Input texts
            
        Returns:
            tuple: (predictions, confidences)
        """
        predictions = []
        confidences = []
        
        # Transform input data
        X_vec = self.vectorizer.transform(X)
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i in range(X_vec.shape[0]):
            # Get non-zero features
            row = X_vec[i].toarray()[0]
            score = 0
            
            # Calculate score based on word weights
            for idx, value in enumerate(row):
                if value > 0:
                    word = feature_names[idx]
                    if word in self.word_weights:
                        score += self.word_weights[word] * value
            
            # Convert score to probability with sigmoid function
            prob_real = 1 / (1 + np.exp(-score))
            
            # Determine prediction and confidence
            if prob_real >= 0.5:
                pred = 1
                conf = prob_real
            else:
                pred = 0
                conf = 1 - prob_real
            
            predictions.append(pred)
            confidences.append(conf)
        
        return np.array(predictions), np.array(confidences)

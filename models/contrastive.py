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
    print("Warning: PyTorch not available. Using simplified implementation for Contrastive Learning.")

# Define PyTorch modules only if torch is available
if TORCH_AVAILABLE:
    class ContrastiveEncoder(nn.Module):
        """Encoder model for contrastive learning."""
        
        def __init__(self, input_dim, embedding_dim=128):
            super(ContrastiveEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embedding_dim)
            )
            
        def forward(self, x):
            return F.normalize(self.encoder(x), dim=1)  # L2 normalize

    class ContrastiveClassifier(nn.Module):
        """Classifier that uses embeddings from the contrastive encoder."""
        
        def __init__(self, embedding_dim, n_classes=2):
            super(ContrastiveClassifier, self).__init__()
            self.classifier = nn.Linear(embedding_dim, n_classes)
            
        def forward(self, x):
            return self.classifier(x)
else:
    # Dummy classes when torch is not available
    class ContrastiveEncoder:
        pass
        
    class ContrastiveClassifier:
        pass

class ContrastiveModel:
    """
    Implementation of Contrastive Learning (inspired by SimCLR, MoCo) for fake news detection.
    
    This model learns representations by pulling together similar examples (same class)
    and pushing apart dissimilar ones (different classes).
    """
    
    def __init__(self, max_features=5000, embedding_dim=128, temperature=0.5):
        """
        Initialize the Contrastive Learning model.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            embedding_dim (int): Dimension of the embedding space
            temperature (float): Temperature parameter for contrastive loss
        """
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Check if torch is available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder = None
            self.classifier = None
            self.is_simplified = False
        else:
            # Use simplified implementation
            self.is_simplified = True
            self.word_vectors = {}
            self.class_centroids = {}
    
    def train(self, X_train, y_train, epochs=3, batch_size=32):
        """
        Train the Contrastive Learning model.
        
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
            self.encoder = ContrastiveEncoder(input_dim, self.embedding_dim).to(self.device)
        
        if self.classifier is None:
            self.classifier = ContrastiveClassifier(self.embedding_dim).to(self.device)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train_vec).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizers
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
        
        # Training loop
        for epoch in range(epochs):
            # Contrastive learning phase
            self.encoder.train()
            contrastive_loss_sum = 0
            
            for batch_X, batch_y in dataloader:
                # Move tensors to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get embeddings
                embeddings = self.encoder(batch_X)
                
                # Calculate contrastive loss
                contrastive_loss = self._contrastive_loss(embeddings, batch_y)
                
                # Update encoder
                encoder_optimizer.zero_grad()
                contrastive_loss.backward()
                encoder_optimizer.step()
                
                contrastive_loss_sum += contrastive_loss.item()
            
            avg_contrastive_loss = contrastive_loss_sum / len(dataloader)
            
            # Classification phase
            self.encoder.eval()
            self.classifier.train()
            classification_loss_sum = 0
            
            for batch_X, batch_y in dataloader:
                # Move tensors to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get embeddings (no gradient needed)
                with torch.no_grad():
                    embeddings = self.encoder(batch_X)
                
                # Forward pass for classification
                logits = self.classifier(embeddings)
                classification_loss = F.cross_entropy(logits, batch_y)
                
                # Update classifier
                classifier_optimizer.zero_grad()
                classification_loss.backward()
                classifier_optimizer.step()
                
                classification_loss_sum += classification_loss.item()
            
            avg_classification_loss = classification_loss_sum / len(dataloader)
            
            print(f"Epoch {epoch+1}/{epochs}, Contrastive Loss: {avg_contrastive_loss:.4f}, Classification Loss: {avg_classification_loss:.4f}")
        
        return self
    
    def _contrastive_loss(self, embeddings, labels):
        """
        Calculate contrastive loss (NT-Xent loss).
        
        Args:
            embeddings (torch.Tensor): Embeddings from the encoder
            labels (torch.Tensor): Class labels
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask_positives = (labels == labels.T).float()
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Set diagonal to large negative value to exclude self-similarity
        logits_mask = torch.ones_like(mask_positives) - torch.eye(mask_positives.shape[0]).to(self.device)
        mask_positives = mask_positives * logits_mask
        
        # Calculate exp sum
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood for positive pairs
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / mask_positives.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _train_simplified(self, X_train, y_train):
        """
        Simplified training method for demonstration when PyTorch is not available.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
        """
        # Use TF-IDF to extract features
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        
        # Get unique classes
        classes = np.unique(y_train)
        
        # Create vector representations for each class
        self.class_centroids = {}
        for cls in classes:
            # Get vectors for this class
            class_vectors = X_train_vec[y_train == cls]
            
            # Compute centroid
            self.class_centroids[cls] = class_vectors.mean(axis=0)
        
        # Scale centroids to unit vectors for cosine similarity
        for cls in self.class_centroids:
            norm = np.linalg.norm(self.class_centroids[cls])
            if norm > 0:
                self.class_centroids[cls] /= norm
    
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
            embeddings = self.encoder(X_tensor)
            logits = self.classifier(embeddings)
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
        # Transform input data
        X_vec = self.vectorizer.transform(X).toarray()
        
        predictions = []
        confidences = []
        
        for i in range(X_vec.shape[0]):
            # Normalize vector
            vec = X_vec[i]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            # Calculate similarities to each class centroid
            similarities = {}
            for cls, centroid in self.class_centroids.items():
                similarities[cls] = np.dot(vec, centroid)
            
            # Get prediction and confidence
            if similarities[1] > similarities[0]:
                pred = 1
                conf = 0.5 + (similarities[1] - similarities[0]) / 2
            else:
                pred = 0
                conf = 0.5 + (similarities[0] - similarities[1]) / 2
            
            # Clip confidence to [0, 1]
            conf = max(0, min(conf, 1))
            
            predictions.append(pred)
            confidences.append(conf)
        
        return np.array(predictions), np.array(confidences)

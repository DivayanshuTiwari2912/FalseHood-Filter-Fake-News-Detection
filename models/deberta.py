import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import transformers and torch, but provide fallbacks if not available
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers or torch packages not available. Using simplified implementation.")

class DeBERTaModel:
    """
    Implementation of DeBERTa (Decoding-Enhanced BERT with Disentangled Attention)
    for fake news detection.
    """
    
    def __init__(self, model_name="microsoft/deberta-base", max_length=256):
        """
        Initialize the DeBERTa model.
        
        Args:
            model_name (str): Name of the pretrained DeBERTa model
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Check if transformers and torch are available
        if TRANSFORMERS_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize model with classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2
            )
            
            # Flag for simplified implementation
            self.is_simplified = False
        else:
            # Use simplified implementation with basic TF-IDF
            self.is_simplified = True
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.word_weights = {}
            
            # No model to move in this case
    
    def train(self, X_train, y_train, epochs=3, batch_size=8):
        """
        Train the DeBERTa model.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            self: Trained model
        """
        # Check if using simplified implementation
        if hasattr(self, 'is_simplified') and self.is_simplified:
            self._train_simplified(X_train, y_train)
            return self
        
        # Convert labels to PyTorch tensors
        labels = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        # Tokenize inputs
        encoded_inputs = self._tokenize_batch(X_train)
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            encoded_inputs['input_ids'],
            encoded_inputs['attention_mask'],
            labels
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set model to training mode
        self.model.train()
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Unpack batch
                input_ids, attention_mask, batch_labels = batch
                
                # Move tensors to device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                # Calculate loss
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def _train_simplified(self, X_train, y_train):
        """
        Simplified training method for demonstration when transformers library is not available.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
        """
        # Count word frequencies for each class
        word_counts = {0: {}, 1: {}}
        class_counts = {0: 0, 1: 0}
        
        for text, label in zip(X_train, y_train):
            class_counts[label] += 1
            words = text.lower().split()
            
            for word in set(words):  # Use set to count each word once per document
                if word not in word_counts[label]:
                    word_counts[label][word] = 0
                word_counts[label][word] += 1
        
        # Calculate word weights using TF-IDF like approach
        self.word_weights = {}
        vocab = set()
        for label_words in word_counts.values():
            vocab.update(label_words.keys())
        
        for word in vocab:
            # Add smoothing to avoid division by zero
            fake_freq = (word_counts[0].get(word, 0) + 1) / (class_counts[0] + 2)
            real_freq = (word_counts[1].get(word, 0) + 1) / (class_counts[1] + 2)
            
            # Log ratio as a simple feature weight
            self.word_weights[word] = np.log(real_freq / fake_freq)
        
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
        if hasattr(self, 'is_simplified') and self.is_simplified:
            return self._predict_simplified(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        predictions = []
        confidences = []
        
        # Process in batches to avoid out of memory issues
        batch_size = 8
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            # Tokenize batch
            encoded_inputs = self._tokenize_batch(batch)
            
            # Move tensors to device
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                
                # Get predicted class and confidence
                batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                batch_confidences = probs.max(dim=1)[0].cpu().numpy()
                
                predictions.extend(batch_preds)
                confidences.extend(batch_confidences)
        
        return np.array(predictions), np.array(confidences)
    
    def _predict_simplified(self, X):
        """
        Simplified prediction method for demonstration when transformers library is not available.
        
        Args:
            X (array-like): Input texts
            
        Returns:
            tuple: (predictions, confidences)
        """
        predictions = []
        confidences = []
        
        for text in X:
            words = text.lower().split()
            
            # Calculate score based on word weights
            score = 0
            for word in words:
                if word in self.word_weights:
                    score += self.word_weights[word]
            
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
    
    def _tokenize_batch(self, texts):
        """
        Tokenize a batch of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Tokenized inputs
        """
        # Check if tokenizer is available
        if self.tokenizer is None:
            return None
        
        # Tokenize
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded_inputs

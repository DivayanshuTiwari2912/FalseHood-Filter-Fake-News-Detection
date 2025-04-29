import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import deque
import random
import pandas as pd

# Try to import torch, but provide fallbacks if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using simplified implementation for RL model.")

# Define PyTorch modules only if torch is available
if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for fake news detection."""
        
        def __init__(self, input_dim, hidden_dim=128, n_actions=2):
            super(DQNNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
            
        def forward(self, x):
            return self.network(x)

    class PolicyNetwork(nn.Module):
        """Policy network for policy gradient methods."""
        
        def __init__(self, input_dim, hidden_dim=128, n_actions=2):
            super(PolicyNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )
            
        def forward(self, x):
            return F.softmax(self.network(x), dim=1)
else:
    # Dummy classes when torch is not available
    class DQNNetwork:
        pass
        
    class PolicyNetwork:
        pass

class RLModel:
    """
    Implementation of Reinforcement Learning methods for fake news detection.
    
    Combines Deep Q-Networks and Policy Gradient methods.
    """
    
    def __init__(self, max_features=5000, hidden_dim=128, method='dqn'):
        """
        Initialize the RL model.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorizer
            hidden_dim (int): Dimension of hidden layers
            method (str): RL method to use ('dqn' or 'policy_gradient')
        """
        self.max_features = max_features
        self.hidden_dim = hidden_dim
        self.method = method
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Check if torch is available
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Models will be initialized during training when input dimension is known
            self.dqn = None
            self.policy_net = None
            
            # Hyperparameters
            self.gamma = 0.99  # Discount factor
            self.epsilon = 0.1  # Exploration rate
            self.replay_buffer = deque(maxlen=1000)  # Experience replay buffer
            self.batch_size = 32
            
            # Flag for simplified implementation
            self.is_simplified = False
        else:
            # Use simplified implementation
            self.is_simplified = True
            self.feature_weights = None
    
    def train(self, X_train, y_train, epochs=3, batch_size=32):
        """
        Train the RL model.
        
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
        if self.dqn is None:
            self.dqn = DQNNetwork(input_dim, self.hidden_dim).to(self.device)
            self.target_dqn = DQNNetwork(input_dim, self.hidden_dim).to(self.device)
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        if self.policy_net is None:
            self.policy_net = PolicyNetwork(input_dim, self.hidden_dim).to(self.device)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train_vec).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizers
        if self.method == 'dqn':
            optimizer = torch.optim.Adam(self.dqn.parameters(), lr=1e-3)
        else:  # policy_gradient
            optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            total_reward = 0
            
            for batch_X, batch_y in dataloader:
                # Move tensors to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if self.method == 'dqn':
                    # DQN training
                    loss = self._train_dqn_step(batch_X, batch_y, optimizer)
                    total_loss += loss
                else:
                    # Policy gradient training
                    loss, reward = self._train_policy_gradient_step(batch_X, batch_y, optimizer)
                    total_loss += loss
                    total_reward += reward
            
            # Update target network for DQN
            if self.method == 'dqn' and epoch % 5 == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            
            if self.method == 'dqn':
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, "
                     f"Average Reward: {total_reward/len(dataloader):.4f}")
        
        return self
    
    def _train_dqn_step(self, states, targets, optimizer):
        """
        Perform a single DQN training step.
        
        Args:
            states (torch.Tensor): Input states (text features)
            targets (torch.Tensor): Target labels
            optimizer: Optimizer for the DQN
            
        Returns:
            float: Loss value
        """
        # Calculate rewards based on correct predictions
        rewards = torch.zeros(len(targets)).to(self.device)
        
        # Q-values for current states
        q_values = self.dqn(states)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            actions = torch.randint(0, 2, (len(targets),)).to(self.device)
        else:
            actions = torch.argmax(q_values, dim=1)
        
        # Reward is 1 for correct prediction, -1 for incorrect
        rewards = torch.where(actions == targets, torch.ones_like(rewards), -torch.ones_like(rewards))
        
        # Store experiences in replay buffer
        for i in range(len(states)):
            self.replay_buffer.append((
                states[i].detach().cpu().numpy(),
                actions[i].item(),
                rewards[i].item(),
                targets[i].item()
            ))
        
        # Skip training if buffer is too small
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch_states = torch.FloatTensor(np.array([exp[0] for exp in batch])).to(self.device)
        batch_actions = torch.LongTensor(np.array([exp[1] for exp in batch])).to(self.device)
        batch_rewards = torch.FloatTensor(np.array([exp[2] for exp in batch])).to(self.device)
        batch_targets = torch.LongTensor(np.array([exp[3] for exp in batch])).to(self.device)
        
        # Current Q-values for batch states
        current_q = self.dqn(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_dqn(batch_states).max(1)[0]
        target_q = batch_rewards + self.gamma * next_q
        
        # Calculate loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _train_policy_gradient_step(self, states, targets, optimizer):
        """
        Perform a single policy gradient training step.
        
        Args:
            states (torch.Tensor): Input states (text features)
            targets (torch.Tensor): Target labels
            optimizer: Optimizer for the policy network
            
        Returns:
            tuple: (loss, reward)
        """
        # Get action probabilities from policy network
        action_probs = self.policy_net(states)
        
        # Sample actions from the probabilities
        m = torch.distributions.Categorical(action_probs)
        actions = m.sample()
        
        # Calculate rewards
        rewards = torch.where(actions == targets, torch.ones(len(targets)), -torch.ones(len(targets))).to(self.device)
        
        # Calculate loss using policy gradient
        loss = -m.log_prob(actions) * rewards
        loss = loss.mean()
        
        # Update network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), rewards.mean().item()
    
    def _train_simplified(self, X_train, y_train):
        """
        Simplified training method for demonstration when PyTorch is not available.
        
        Args:
            X_train (array-like): Training texts
            y_train (array-like): Training labels
        """
        # Use TF-IDF to extract features
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        
        # Initialize weights randomly
        self.feature_weights = np.random.uniform(-0.01, 0.01, X_train_vec.shape[1])
        
        # Simple gradient-based update
        learning_rate = 0.01
        n_iterations = 100
        
        for _ in range(n_iterations):
            # Calculate predictions
            scores = X_train_vec.dot(self.feature_weights)
            predictions = (scores > 0).astype(int)
            
            # Calculate errors
            errors = y_train - predictions
            
            # Update weights
            gradient = X_train_vec.T.dot(errors) / len(X_train)
            self.feature_weights += learning_rate * gradient
    
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
            
        # Ensure X is iterable and contains valid text
        if not isinstance(X, (list, tuple, np.ndarray)):
            X = [str(X) if X is not None else ""]
        
        # Convert all inputs to strings, handling possible NaN values
        safe_X = []
        for x in X:
            try:
                if pd.isna(x):
                    safe_X.append("")
                else:
                    safe_X.append(str(x))
            except Exception as e:
                print(f"Error converting input to string: {str(e)}")
                safe_X.append("")
        
        try:
            # Transform input data
            X_vec = self.vectorizer.transform(safe_X).toarray()
            X_tensor = torch.FloatTensor(X_vec).to(self.device)
            
            # Set models to evaluation mode
            if self.method == 'dqn':
                if self.dqn is None:
                    print("DQN model not initialized")
                    return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
                    
                self.dqn.eval()
                try:
                    with torch.no_grad():
                        q_values = self.dqn(X_tensor)
                        preds = torch.argmax(q_values, dim=1).cpu().numpy()
                        
                        # Calculate confidence as normalized absolute difference between Q-values
                        q_diff = torch.abs(q_values[:, 1] - q_values[:, 0])
                        confs = (q_diff / (torch.max(q_diff) + 1e-10)).cpu().numpy()
                except Exception as e:
                    print(f"Error in DQN prediction: {str(e)}")
                    return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
            else:  # policy_gradient
                if self.policy_net is None:
                    print("Policy network not initialized")
                    return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
                    
                self.policy_net.eval()
                try:
                    with torch.no_grad():
                        action_probs = self.policy_net(X_tensor)
                        preds = torch.argmax(action_probs, dim=1).cpu().numpy()
                        confs = action_probs.max(dim=1)[0].cpu().numpy()
                except Exception as e:
                    print(f"Error in policy gradient prediction: {str(e)}")
                    return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
            
            return preds, confs
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
    
    def _predict_simplified(self, X):
        """
        Simplified prediction method for demonstration.
        
        Args:
            X (array-like): Input texts
            
        Returns:
            tuple: (predictions, confidences)
        """
        # Ensure X is iterable and contains valid text
        if not isinstance(X, (list, tuple, np.ndarray)):
            X = [str(X) if X is not None else ""]
        
        # Convert all inputs to strings, handling possible NaN values
        safe_X = []
        for x in X:
            try:
                if pd.isna(x):
                    safe_X.append("")
                else:
                    safe_X.append(str(x))
            except Exception as e:
                print(f"Error converting input to string: {str(e)}")
                safe_X.append("")
        
        try:
            # Transform input data
            X_vec = self.vectorizer.transform(safe_X).toarray()
            
            # Check if feature weights are available
            if self.feature_weights is None or len(self.feature_weights) != X_vec.shape[1]:
                print("Feature weights not properly initialized or dimension mismatch")
                return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)
            
            # Calculate scores
            scores = X_vec.dot(self.feature_weights)
            
            # Get predictions
            predictions = (scores > 0).astype(int)
            
            # Calculate confidences with safeguards
            # Use clipping to prevent overflow in exp
            clipped_scores = np.clip(np.abs(scores), 0, 20)  # Prevent overflow in exp
            confidences = 1 / (1 + np.exp(-clipped_scores))
            
            return predictions, confidences
            
        except Exception as e:
            print(f"Error in simplified prediction: {str(e)}")
            # Return default predictions in case of error
            return np.zeros(len(safe_X), dtype=int), np.full(len(safe_X), 0.5)

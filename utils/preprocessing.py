import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
# Always download required resources to ensure they are available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Verify resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError as e:
    print(f"Error finding NLTK resources: {e}")
    # Try downloading again with specific path
    nltk.download('stopwords', download_dir='/home/runner/nltk_data')
    nltk.download('wordnet', download_dir='/home/runner/nltk_data')
    nltk.download('punkt', download_dir='/home/runner/nltk_data')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text by removing special characters, lowercasing,
    removing stopwords, and lemmatizing.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize - Simple fallback if NLTK fails
        try:
            tokens = nltk.word_tokenize(text)
        except Exception as e:
            print(f"Error in NLTK word_tokenize: {e}")
            # Simple fallback tokenization
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        try:
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        except Exception as e:
            print(f"Error in stopword removal or lemmatization: {e}")
            # Simple fallback without lemmatization or stopword removal
            pass
        
        # Join tokens back to text
        processed_text = ' '.join(tokens)
        
        return processed_text
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        # Return original text if all else fails
        return text.lower() if isinstance(text, str) else ""

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        X: Feature data
        y: Target labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def load_and_preprocess_csv(file_path, text_col='text', label_col='label'):
    """
    Load and preprocess data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        text_col (str): Name of the column containing text data
        label_col (str): Name of the column containing labels
        
    Returns:
        tuple: (preprocessed_texts, labels)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns '{text_col}' or '{label_col}' not found in the CSV file")
    
    # Preprocess texts
    texts = df[text_col].values
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Get labels
    labels = df[label_col].values
    
    return preprocessed_texts, labels

def tokenize_texts(texts, max_length=256):
    """
    Tokenize a list of texts into word indices.
    
    Args:
        texts (list): List of text strings
        max_length (int): Maximum sequence length
        
    Returns:
        list: List of tokenized texts
    """
    # This is a placeholder function
    # In a real implementation, you would use a proper tokenizer from a library
    # like transformers or tokenizers
    
    # For the sake of this example, we'll just split by spaces and pad/truncate
    tokenized = []
    for text in texts:
        tokens = text.split()[:max_length]  # Truncate
        if len(tokens) < max_length:
            tokens += [''] * (max_length - len(tokens))  # Pad
        tokenized.append(tokens)
    
    return tokenized

# About Falsehood Filter

## Project Overview

Falsehood Filter is an advanced false information detection application that leverages state-of-the-art machine learning algorithms to identify misleading or false information. The system analyzes news text and provides predictions on whether the content is likely to be false or real.

## Features

- **Multi-algorithm Analysis**: Uses a combination of four advanced ML algorithms to provide comprehensive analysis
- **Custom Training**: Allows users to train models on their own datasets
- **Performance Comparison**: Visualizes and compares performance metrics across algorithms
- **Detailed Analysis**: Provides confidence scores and explanations for predictions
- **Export Functionality**: Export results and model performance metrics

## Algorithms

The application implements the following algorithms:

### 1. DeBERTa (Decoding-Enhanced BERT with Disentangled Attention)

DeBERTa is an advanced transformer model that enhances BERT by using a disentangled attention mechanism and an enhanced mask decoder. This allows the model to better capture the nuanced relationships between words in text.

Key features:
- Disentangled attention on content and position
- Enhanced mask decoder
- Pre-trained on large text corpora

### 2. Model-Agnostic Meta-Learning (MAML)

MAML is a meta-learning algorithm that trains a model on a variety of learning tasks so that it can quickly adapt to new tasks with minimal data. In our application, it helps the model quickly adapt to different types of false information content.

Key features:
- Fast adaptation to new tasks
- Requires fewer examples to learn
- Optimizes for adaptability

### 3. Contrastive Learning (SimCLR, MoCo)

Contrastive learning techniques like SimCLR and MoCo help the model learn by comparing similar and dissimilar examples. These algorithms create robust representations by pulling similar examples closer together in the embedding space while pushing dissimilar ones apart.

Key features:
- Learns meaningful text representations
- Improves performance with limited labeled data
- Creates more robust models

### 4. Deep Q-Networks (DQN) & Policy Gradient Methods

Reinforcement learning approaches adapted for text classification. The model learns to make classification decisions by maximizing a reward signal (correct classifications).

Key features:
- Learns from feedback through rewards
- Adapts to changing patterns over time
- Balances exploration and exploitation

## How It Works

1. **Data Processing**: Text is preprocessed, tokenized, and transformed into numerical representations
2. **Feature Extraction**: Each algorithm extracts relevant features for classification
3. **Model Training**: Models learn patterns from labeled examples (real vs. false)
4. **Prediction**: New text is analyzed by the trained models
5. **Ensemble Decision**: Results from multiple algorithms are combined for the final verdict

## Resources

To learn more about false information detection and the algorithms used in this application, check out these resources:

- [Transformer Models in NLP](https://huggingface.co/transformers/)
- [Meta-Learning Introduction](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
- [Contrastive Learning](https://arxiv.org/abs/2004.11362)
- [Reinforcement Learning for NLP](https://arxiv.org/abs/1811.06526)

---

Developed as part of a project exploring advanced machine learning approaches to misinformation detection.

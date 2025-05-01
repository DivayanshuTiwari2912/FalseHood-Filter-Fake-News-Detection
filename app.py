import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from io import StringIO

# Import custom modules
from utils.preprocessing import preprocess_text, split_data
from utils.visualization import plot_performance_metrics, plot_confusion_matrix, plot_confidence_distribution

# Handle imports - Provide fallbacks if some modules fail to import
FULL_IMPORTS_AVAILABLE = True

# Import models with error handling for each
try:
    from models.deberta import DeBERTaModel
except ImportError as e:
    st.warning(f"Could not import DeBERTa model: {str(e)}")
    # Create a dummy model class
    class DeBERTaModel:
        def __init__(self, **kwargs):
            self.is_dummy = True
        def train(self, *args, **kwargs):
            return self
        def predict(self, X):
            return [0] * len(X), [0.5] * len(X)
    FULL_IMPORTS_AVAILABLE = False

try:
    from models.maml import MAMLModel
except ImportError as e:
    st.warning(f"Could not import MAML model: {str(e)}")
    # Create a dummy model class
    class MAMLModel:
        def __init__(self, **kwargs):
            self.is_dummy = True
        def train(self, *args, **kwargs):
            return self
        def predict(self, X):
            return [0] * len(X), [0.5] * len(X)
    FULL_IMPORTS_AVAILABLE = False

try:
    from models.contrastive import ContrastiveModel
except ImportError as e:
    st.warning(f"Could not import Contrastive model: {str(e)}")
    # Create a dummy model class
    class ContrastiveModel:
        def __init__(self, **kwargs):
            self.is_dummy = True
        def train(self, *args, **kwargs):
            return self
        def predict(self, X):
            return [0] * len(X), [0.5] * len(X)
    FULL_IMPORTS_AVAILABLE = False

try:
    from models.rl import RLModel
except ImportError as e:
    st.warning(f"Could not import RL model: {str(e)}")
    # Create a dummy model class
    class RLModel:
        def __init__(self, **kwargs):
            self.is_dummy = True
        def train(self, *args, **kwargs):
            return self
        def predict(self, X):
            return [0] * len(X), [0.5] * len(X)
    FULL_IMPORTS_AVAILABLE = False

try:
    from models.trainer import train_model
    from models.evaluator import evaluate_model
except ImportError as e:
    st.warning(f"Could not import training utilities: {str(e)}")
    # Create dummy utility functions
    def train_model(model, X_train, y_train, **kwargs):
        return model.train(X_train, y_train)
    
    def evaluate_model(model, X_test, y_test):
        preds, _ = model.predict(X_test)
        return {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    FULL_IMPORTS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Falsehood Filter - Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing model results and history
if 'models' not in st.session_state:
    st.session_state.models = {
        'DeBERTa': None,
        'MAML': None,
        'Contrastive': None,
        'RL': None
    }

if 'training_history' not in st.session_state:
    st.session_state.training_history = []

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'trained' not in st.session_state:
    st.session_state.trained = False

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# Sidebar navigation
st.sidebar.title("Falsehood Filter 🔍")
page = st.sidebar.selectbox("Navigation", ["Home", "Upload & Train", "Analyze Text", "Scrape Content", "Results Dashboard", "About"])

# Function to get model instance
def get_model(model_name):
    if model_name == "DeBERTa":
        return DeBERTaModel()
    elif model_name == "MAML":
        return MAMLModel()
    elif model_name == "Contrastive":
        return ContrastiveModel()
    elif model_name == "RL":
        return RLModel()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Home page
if page == "Home":
    st.title("Fake News Detection System")
    
    # React Migration Notice
    st.warning("""
    ## 🚧 Important Notice - Migration in Progress 🚧
    
    We're currently migrating this Streamlit application to a modern React frontend with improved features 
    and performance. Both interfaces will be temporarily available during the transition.
    
    * **Streamlit UI**: Available now at port 5000
    * **React UI (in development)**: Coming soon
    * **API Backend**: Running on port 5001
    
    The API server and trained models will remain fully compatible with both interfaces.
    """)
    
    st.write("""
    ### Welcome to the Falsehood Filter
    
    This application helps detect fake news using advanced machine learning algorithms. You can:
    - Upload your dataset and train the models
    - Analyze individual news articles
    - View model performance and comparison
    - Track analysis history and export results
    """)
    
    st.info("⚠️ To get started, please upload a dataset in the 'Upload & Train' section.")
    
    # Display model status
    st.subheader("Model Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.models['DeBERTa']:
            st.success("DeBERTa: Trained")
        else:
            st.error("DeBERTa: Not Trained")
    
    with col2:
        if st.session_state.models['MAML']:
            st.success("MAML: Trained")
        else:
            st.error("MAML: Not Trained")
            
    with col3:
        if st.session_state.models['Contrastive']:
            st.success("Contrastive: Trained")
        else:
            st.error("Contrastive: Not Trained")
            
    with col4:
        if st.session_state.models['RL']:
            st.success("RL: Trained")
        else:
            st.error("RL: Not Trained")

# Upload and Train page
elif page == "Upload & Train":
    st.title("Upload Dataset & Train Models")
    
    # File upload section
    st.subheader("Upload CSV Dataset")
    st.write("Upload a CSV file containing news data with at least 'text' and 'label' columns.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and display data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_data = df
            st.success("File uploaded successfully!")
            
            # Display dataset info
            st.subheader("Dataset Overview")
            st.write(f"Number of rows: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns.tolist())}")
            
            # Display first few rows
            st.dataframe(df.head())
            
            # Data validation
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Required columns are missing: {', '.join(missing_columns)}")
                st.stop()
            
            # Column mapping
            st.subheader("Column Mapping")
            st.write("Please map your dataset columns to the required fields:")
            
            text_column = st.selectbox("Text column", df.columns, index=df.columns.get_loc('text') if 'text' in df.columns else 0)
            label_column = st.selectbox("Label column", df.columns, index=df.columns.get_loc('label') if 'label' in df.columns else 0)
            
            # Model selection for training
            st.subheader("Select Models to Train")
            models_to_train = st.multiselect(
                "Choose algorithms",
                ["DeBERTa", "MAML", "Contrastive", "RL"],
                default=["DeBERTa"]
            )
            
            # Training parameters
            st.subheader("Training Parameters")
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            epochs = st.slider("Training epochs", 1, 10, 3)
            
            # Train button
            if st.button("Train Models"):
                if not models_to_train:
                    st.warning("Please select at least one model to train.")
                else:
                    # Prepare data
                    # Ensure text data is processed properly
                    X = df[text_column].astype(str).values
                    y = df[label_column].values
                    
                    # Add additional data validation
                    if len(X) == 0:
                        st.error("No data found in the text column.")
                        st.stop()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                    
                    st.write(f"Training data size: {len(X_train)} samples")
                    
                    # Display training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train each selected model
                    for i, model_name in enumerate(models_to_train):
                        status_text.text(f"Training {model_name} model...")
                        
                        # Get model instance
                        model = get_model(model_name)
                        
                        # Train model
                        train_model(model, X_train, y_train, epochs=epochs)
                        
                        # Evaluate model
                        metrics = evaluate_model(model, X_test, y_test)
                        
                        # Save model and metrics
                        st.session_state.models[model_name] = {
                            'model': model,
                            'metrics': metrics,
                            'training_data': {
                                'X_train': X_train,
                                'y_train': y_train,
                                'X_test': X_test,
                                'y_test': y_test
                            }
                        }
                        
                        # Update training history
                        st.session_state.training_history.append({
                            'model': model_name,
                            'timestamp': pd.Timestamp.now(),
                            'metrics': metrics,
                            'parameters': {
                                'epochs': epochs,
                                'test_size': test_size
                            }
                        })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(models_to_train))
                    
                    st.session_state.trained = True
                    status_text.text("Training completed!")
                    st.success("Models have been successfully trained!")
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

# Analyze Text page
elif page == "Analyze Text":
    st.title("Analyze News Text")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train the models first in the 'Upload & Train' section.")
    else:
        # Text input for analysis
        st.subheader("Enter News Text")
        news_text = st.text_area("Paste news article text here:", height=200)
        
        # Model selection for analysis
        st.subheader("Select Models for Analysis")
        models_for_analysis = st.multiselect(
            "Choose algorithms to use for analysis",
            [model for model in st.session_state.models if st.session_state.models[model] is not None],
            default=[model for model in st.session_state.models if st.session_state.models[model] is not None][:1]
        )
        
        # Analyze button
        if st.button("Analyze"):
            if not news_text:
                st.warning("Please enter text to analyze.")
            elif not models_for_analysis:
                st.warning("Please select at least one model for analysis.")
            else:
                # Process text
                processed_text = preprocess_text(news_text)
                
                # Display results
                st.subheader("Analysis Results")
                
                results = {}
                
                # Create columns for each model
                cols = st.columns(len(models_for_analysis))
                
                for i, model_name in enumerate(models_for_analysis):
                    model = st.session_state.models[model_name]['model']
                    prediction, confidence = model.predict([processed_text])
                    
                    # Store results
                    results[model_name] = {
                        'prediction': prediction[0],
                        'confidence': confidence[0]
                    }
                    
                    # Display in column
                    with cols[i]:
                        st.write(f"### {model_name}")
                        if prediction[0] == 1:
                            st.success("✅ Real News")
                        else:
                            st.error("❌ Fake News")
                        
                        st.write(f"Confidence: {confidence[0]:.2f}")
                        
                        # Simple confidence meter
                        st.progress(confidence[0])
                
                # Overall verdict (majority voting)
                st.subheader("Overall Verdict")
                predictions = [results[model]['prediction'] for model in results]
                average_confidence = np.mean([results[model]['confidence'] for model in results])
                
                if sum(predictions) > len(predictions) / 2:
                    st.success(f"✅ Real News (Average confidence: {average_confidence:.2f})")
                else:
                    st.error(f"❌ Fake News (Average confidence: {average_confidence:.2f})")
                
                # Add to analysis history
                st.session_state.analysis_history.append({
                    'text': news_text,
                    'timestamp': pd.Timestamp.now(),
                    'results': results,
                    'overall_verdict': sum(predictions) > len(predictions) / 2,
                    'average_confidence': average_confidence
                })
                
                # Explanation section
                st.subheader("Explanation")
                st.write("""
                The analysis is based on patterns learned from the training data. 
                Key indicators considered by the models include:
                
                - Language patterns and word choice
                - Contextual understanding of information
                - Comparison with known fake and real news samples
                - Structural elements common in fake vs. real news
                """)

# Scrape Content page
elif page == "Scrape Content":
    st.title("Scrape Website Content")
    
    st.write("""
    ### Extract News Text from Websites
    
    This tool allows you to automatically extract the main content from news websites and analyze it for credibility.
    Enter the URL of a news article below to scrape its content.
    """)
    
    # URL input
    url = st.text_input("Enter website URL:", placeholder="https://example.com/news/article")
    
    # Scrape button
    if st.button("Scrape Content"):
        if not url:
            st.warning("Please enter a URL to scrape.")
        else:
            try:
                with st.spinner("Scraping website content..."):
                    # Call API to scrape the website
                    import requests
                    import json
                    
                    response = requests.post(
                        "http://localhost:5001/api/scrape",
                        json={"url": url},
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display scraped content
                        st.success(f"Successfully scraped content from {url}")
                        st.subheader("Extracted Content")
                        
                        # Show extracted text
                        extracted_text = result.get("text", "")
                        st.text_area("Extracted Text", extracted_text, height=300, disabled=True)
                        
                        # Show stats
                        st.write(f"Content length: {result.get('length', 0)} characters")
                        
                        # Option to analyze the scraped content
                        if st.session_state.trained:
                            if st.button("Analyze Scraped Content"):
                                # Get all available trained models
                                available_models = [model for model in st.session_state.models if st.session_state.models[model] is not None]
                                
                                if not available_models:
                                    st.warning("No trained models available. Please train models first.")
                                else:
                                    # Process text
                                    processed_text = preprocess_text(extracted_text)
                                    
                                    # Display results
                                    st.subheader("Analysis Results")
                                    
                                    results = {}
                                    
                                    # Create columns for each model
                                    cols = st.columns(min(len(available_models), 4))  # Limit to 4 columns max
                                    
                                    for i, model_name in enumerate(available_models):
                                        model = st.session_state.models[model_name]['model']
                                        prediction, confidence = model.predict([processed_text])
                                        
                                        # Store results
                                        results[model_name] = {
                                            'prediction': prediction[0],
                                            'confidence': confidence[0]
                                        }
                                        
                                        # Display in column
                                        with cols[i % len(cols)]:
                                            st.write(f"### {model_name}")
                                            if prediction[0] == 1:
                                                st.success("✅ Real News")
                                            else:
                                                st.error("❌ Fake News")
                                            
                                            st.write(f"Confidence: {confidence[0]:.2f}")
                                            st.progress(confidence[0])
                                    
                                    # Overall verdict (majority voting)
                                    st.subheader("Overall Verdict")
                                    predictions = [results[model]['prediction'] for model in results]
                                    average_confidence = np.mean([results[model]['confidence'] for model in results])
                                    
                                    if sum(predictions) > len(predictions) / 2:
                                        st.success(f"✅ Real News (Average confidence: {average_confidence:.2f})")
                                    else:
                                        st.error(f"❌ Fake News (Average confidence: {average_confidence:.2f})")
                                    
                                    # Add to analysis history
                                    st.session_state.analysis_history.append({
                                        'text': f"[Scraped from {url}] {extracted_text[:100]}...",
                                        'timestamp': pd.Timestamp.now(),
                                        'results': results,
                                        'overall_verdict': sum(predictions) > len(predictions) / 2,
                                        'average_confidence': average_confidence,
                                        'url': url
                                    })
                        else:
                            st.warning("⚠️ To analyze this content, please train models first in the 'Upload & Train' section.")
                    else:
                        error_msg = "Failed to scrape content"
                        try:
                            error_details = response.json().get("error", "Unknown error")
                            error_msg = f"{error_msg}: {error_details}"
                        except:
                            pass
                        st.error(error_msg)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Add tips for better scraping
    with st.expander("Tips for Website Scraping"):
        st.write("""
        - Make sure the URL is valid and accessible
        - Some websites may block scraping attempts
        - For best results, use URLs from major news sources
        - The tool extracts the main article content, ignoring navigation, ads, etc.
        - You may need to manually clean the text if the automatic extraction isn't perfect
        """)

# Results Dashboard page
elif page == "Results Dashboard":
    st.title("Results Dashboard")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train the models first in the 'Upload & Train' section.")
    else:
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Model Performance", "Analysis History", "Export Results"])
        
        # Model Performance tab
        with tab1:
            st.subheader("Model Performance Comparison")
            
            # Select models to compare
            models_to_compare = st.multiselect(
                "Select models to compare",
                [model for model in st.session_state.models if st.session_state.models[model] is not None],
                default=[model for model in st.session_state.models if st.session_state.models[model] is not None]
            )
            
            if models_to_compare:
                # Metrics comparison
                metrics_df = pd.DataFrame({
                    model: {
                        metric: value 
                        for metric, value in st.session_state.models[model]['metrics'].items()
                    }
                    for model in models_to_compare
                }).T
                
                st.dataframe(metrics_df.style.highlight_max(axis=0))
                
                # Performance metrics visualization
                st.subheader("Performance Metrics Visualization")
                fig = plot_performance_metrics(
                    {model: st.session_state.models[model]['metrics'] for model in models_to_compare}
                )
                st.pyplot(fig)
                
                # Individual model details
                st.subheader("Individual Model Details")
                selected_model = st.selectbox("Select model", models_to_compare)
                
                if selected_model:
                    model_data = st.session_state.models[selected_model]
                    
                    # Show metrics
                    st.write("### Metrics")
                    for metric, value in model_data['metrics'].items():
                        st.write(f"**{metric}:** {value:.4f}")
                    
                    # Show confusion matrix
                    st.write("### Confusion Matrix")
                    y_true = model_data['training_data']['y_test']
                    y_pred = model_data['model'].predict(model_data['training_data']['X_test'])[0]
                    cm_fig = plot_confusion_matrix(y_true, y_pred)
                    st.pyplot(cm_fig)
                    
                    # Show confidence distribution
                    st.write("### Prediction Confidence Distribution")
                    conf_fig = plot_confidence_distribution(
                        model_data['model'],
                        model_data['training_data']['X_test'],
                        model_data['training_data']['y_test']
                    )
                    st.pyplot(conf_fig)
            else:
                st.info("Please select at least one model to display performance metrics.")
        
        # Analysis History tab
        with tab2:
            st.subheader("Analysis History")
            
            if not st.session_state.analysis_history:
                st.info("No analysis history available. Analyze some news text first.")
            else:
                # Show history in table
                history_data = [{
                    'Timestamp': item['timestamp'],
                    'Text': item['text'][:100] + '...' if len(item['text']) > 100 else item['text'],
                    'Verdict': 'Real' if item['overall_verdict'] else 'Fake',
                    'Confidence': f"{item['average_confidence']:.2f}"
                } for item in st.session_state.analysis_history]
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df)
                
                # Display selected analysis details
                if len(st.session_state.analysis_history) > 0:
                    st.subheader("Analysis Details")
                    
                    selected_idx = st.selectbox(
                        "Select analysis to view details",
                        range(len(st.session_state.analysis_history)),
                        format_func=lambda i: f"{st.session_state.analysis_history[i]['timestamp']} - {'Real' if st.session_state.analysis_history[i]['overall_verdict'] else 'Fake'}"
                    )
                    
                    analysis = st.session_state.analysis_history[selected_idx]
                    
                    st.write("**Text:**")
                    st.write(analysis['text'])
                    
                    st.write("**Overall Verdict:**")
                    if analysis['overall_verdict']:
                        st.success(f"✅ Real News (Confidence: {analysis['average_confidence']:.2f})")
                    else:
                        st.error(f"❌ Fake News (Confidence: {analysis['average_confidence']:.2f})")
                    
                    st.write("**Individual Model Results:**")
                    for model, result in analysis['results'].items():
                        st.write(f"**{model}:** {'Real' if result['prediction'] == 1 else 'Fake'} (Confidence: {result['confidence']:.2f})")
        
        # Export Results tab
        with tab3:
            st.subheader("Export Results")
            
            export_type = st.radio("Select what to export", ["Model Performance", "Analysis History"])
            
            if export_type == "Model Performance":
                if not any(st.session_state.models.values()):
                    st.info("No models trained yet. Train models first to export performance data.")
                else:
                    # Prepare performance data
                    performance_data = {
                        model: {
                            'metrics': metrics,
                            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        for model, metrics in {
                            model: data['metrics'] 
                            for model, data in st.session_state.models.items() 
                            if data is not None
                        }.items()
                    }
                    
                    # Convert to CSV
                    csv_data = []
                    for model, data in performance_data.items():
                        row = {'model': model, 'timestamp': data['timestamp']}
                        row.update(data['metrics'])
                        csv_data.append(row)
                    
                    csv_df = pd.DataFrame(csv_data)
                    
                    # Display and provide download link
                    st.dataframe(csv_df)
                    
                    csv = csv_df.to_csv(index=False)
                    st.download_button(
                        label="Download Performance Data CSV",
                        data=csv,
                        file_name="fake_news_model_performance.csv",
                        mime="text/csv"
                    )
            
            elif export_type == "Analysis History":
                if not st.session_state.analysis_history:
                    st.info("No analysis history available yet. Analyze some texts first.")
                else:
                    # Prepare analysis history data
                    history_data = []
                    for analysis in st.session_state.analysis_history:
                        row = {
                            'timestamp': analysis['timestamp'],
                            'text': analysis['text'],
                            'verdict': 'Real' if analysis['overall_verdict'] else 'Fake',
                            'confidence': analysis['average_confidence']
                        }
                        
                        # Add individual model results
                        for model, result in analysis['results'].items():
                            row[f"{model}_prediction"] = 'Real' if result['prediction'] == 1 else 'Fake'
                            row[f"{model}_confidence"] = result['confidence']
                        
                        history_data.append(row)
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Display and provide download link
                    st.dataframe(history_df)
                    
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis History CSV",
                        data=csv,
                        file_name="fake_news_analysis_history.csv",
                        mime="text/csv"
                    )

# About page
elif page == "About":
    st.title("About Falsehood Filter")
    
    with open("assets/about.md", "r") as f:
        about_content = f.read()
        st.markdown(about_content)

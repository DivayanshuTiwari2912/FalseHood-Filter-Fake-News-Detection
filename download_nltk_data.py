import nltk
import ssl
import os

# Create custom directory if it doesn't exist
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

try:
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK packages
    for package in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            
    print("NLTK data download complete.")
except Exception as e:
    print(f"Error setting up NLTK: {str(e)}")
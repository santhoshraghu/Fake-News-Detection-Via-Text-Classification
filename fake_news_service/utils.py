import re
import string
import pickle
import os
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_text(text: str) -> str:
    # Initializing lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets  and their contents
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text) # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub('\n', '', text) # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove digits
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', text) # Remove remaining punctuation (excluding hyphens and apostrophes)
    
    # Remove stopwords and lemmatize
    text = [word for word in text.split() if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    
    return text


def save_model(model, vectorizer, model_path: str = "models/model.pkl", 
               vectorizer_path: str = "models/vectorizer.pkl"):
    """
    Save trained model and vectorizer to disk.
    
    Args:
        model: Trained model
        vectorizer: Trained vectorizer
        model_path & vectorizer_path: Paths to save model and vectorizer
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_model(model_path: str = "models/model.pkl", 
               vectorizer_path: str = "models/vectorizer.pkl") -> Tuple:
    """
    Load trained model and vectorizer from disk.
    
    Args:
        model_path & vectorizer_path: Paths to model and vectorizer files
        
    Returns:
        Tuple of (model, vectorizer)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_text(model, vectorizer, text: str) -> str:
    """
    Predict if text is fake or real news.
    
    Args:
        model & vectorizer: Trained model and vectorizer
        text: Text to predict
        
    Returns:
        Prediction result ("FAKE" or "REAL")
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Transform text using vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    
    return "FAKE" if prediction == 0 else "REAL"
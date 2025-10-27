from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple
from .data_loader import load_data, prepare_train_test_data
from .utils import save_model


def train_model(fake_path: str = "fake.csv", true_path: str = "true.csv",
                model_path: str = "models/model.pkl", 
                vectorizer_path: str = "models/vectorizer.pkl") -> Tuple:
    """
    Training a fake news detection model.
    
    Args:
        fake_path & true_path: Path to fake and true news CSV files
        model_path & vectorizer_path: Paths to save trained model and vectorizer

    Returns:
        Tuple of (model, vectorizer, accuracy)
    """
    
    print("Loading and preprocessing data")
    
    # Load and prepare data
    df = load_data(fake_path, true_path)
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize vectorizer and model (using TF-IDF and Logistic Regression)
    vectorizer = TfidfVectorizer()
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    print("Training model...")
    
    X_train_vectorized = vectorizer.fit_transform(X_train) # Transform training data
    
    model.fit(X_train_vectorized, y_train) # Train model
    
    print("Evaluating model...")
    
    # Evaluate model
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    save_model(model, vectorizer, model_path, vectorizer_path)
    
    return model, vectorizer, accuracy
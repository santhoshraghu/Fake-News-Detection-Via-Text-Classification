import sys
import os


# Adding parent directory to path to import fake_news_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_news_service.model import train_model

def main():
    """Main training function."""
    print("FAKE NEWS DETECTION MODEL TRAINING")
    
    # Check if data files exist
    if not os.path.exists("fake.csv"):
        print("Error: fake.csv not found in current directory")
        print("Please ensure fake.csv is in the root directory")
        return
    
    if not os.path.exists("true.csv"):
        print("Error: true.csv not found in current directory")
        print("Please ensure true.csv is in the root directory")
        return
    
    print("Starting to train")
    
    try:
        # Train model
        model, vectorizer, accuracy = train_model()
        
        print("Training Completed!")
        print(f"Model accuracy: {accuracy:.4f}")
        print("Model and vectorizer saved to models/ directory")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please check your files and try again.")


if __name__ == "__main__":
    main()
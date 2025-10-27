import pytest
from fake_news_service.utils import clean_text, predict_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TestUtils:
    """Test class for utility functions."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        test_text = "This is a test! With punctuation, numbers123, and URLs https://example.com"
        cleaned = clean_text(test_text)
        
        # Check that punctuation, numbers, and URLs are removed
        assert "!" not in cleaned
        assert "123" not in cleaned
        assert "https://example.com" not in cleaned
        assert "test" in cleaned.lower()
    
    def test_clean_text_empty(self):
        """Test text cleaning with empty string."""
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_html_tags(self):
        """Test text cleaning with HTML tags."""
        test_text = "<p>This is <b>bold</b> text</p>"
        cleaned = clean_text(test_text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "</p>" not in cleaned
        assert "bold" in cleaned
    
    def test_predict_text_fake(self):
        """Test prediction with fake news text."""
        vectorizer = TfidfVectorizer()
        model = LogisticRegression(random_state=42)
        
        # Actual training examples
        X_train = [
            "Drunk Bragging Trump Staffer Started Russian Collusion Investigation",
            "Breaking: Scientists discover shocking conspiracy about water",
            "WASHINGTON (Reuters) - Federal Reserve announces policy changes",
            "U.S. military to accept new recruits starting Monday"
        ]
        y_train = [0, 0, 1, 1]  # 0 = fake, 1 = real
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        model.fit(X_train_vectorized, y_train)
        
        # Testing with fake news style headline
        result = predict_text(model, vectorizer, "Drunk Bragging Trump Staffer Started Russian Collusion Investigation")
        assert result == "FAKE"
    
    def test_predict_text_real(self):
        """Test prediction with real news text."""
        vectorizer = TfidfVectorizer()
        model = LogisticRegression(random_state=42)
        
        # Actual training examples
        X_train = [
            "WASHINGTON (Reuters) - Federal Reserve announces policy changes",
            "U.S. military to accept new recruits starting Monday",
            "Drunk Bragging Trump Staffer Started Russian Collusion Investigation",
            "Breaking: Scientists discover shocking conspiracy"
        ]
        y_train = [1, 1, 0, 0]  # 1 = real, 0 = fake
        
        X_train_vectorized = vectorizer.fit_transform(X_train)
        model.fit(X_train_vectorized, y_train)
        
        # Test with formal Reuters-style headline
        result = predict_text(model, vectorizer, "WASHINGTON (Reuters) - Federal Reserve announces policy changes")
        assert result == "REAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
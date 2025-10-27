import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import Tuple
from .utils import clean_text


def load_data(fake_path: str = "fake.csv", true_path: str = "true.csv") -> pd.DataFrame:
    """
    Load and combine fake and true news datasets.
    
    Args:
        fake_path & true_path: Path to fake and true news CSV files

    Returns:
        Combined DataFrame with labels
    """
    # Loading datasets
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(true_path)
    
    # Adding labels (0 for fake, 1 for real)
    df_fake['label'] = 0
    df_real['label'] = 1
    
    # Combining title and text into a single News column
    df_fake['News'] = df_fake['title'] + ' ' + df_fake['text']
    df_real['News'] = df_real['title'] + ' ' + df_real['text']
    
    # Removing duplicates
    df_fake.drop_duplicates(inplace=True)
    df_real.drop_duplicates(inplace=True)
    
    # Combining datasets
    df_combined = pd.concat([df_real, df_fake], ignore_index=True)
    
    # Cleaing text data using function from utils ( Lematization, removing stopwords, special characters etc.)
    df_combined['News'] = df_combined['News'].apply(clean_text)
    
    return df_combined


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.3, 
                           random_state: int = 42) -> Tuple:
    """
    Preparing training and testing data.
    
    Args:
        df: Combined DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Shuffling the data for random distribution
    df_shuffled = shuffle(df, random_state=random_state)
    
    # Splitting features and labels
    X = df_shuffled['News']
    y = df_shuffled['label']
    
    # Splitting into train and test sets (70% train, 30% test as in notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
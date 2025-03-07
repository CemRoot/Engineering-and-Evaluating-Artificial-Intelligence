import numpy as np
import pandas as pd
from Config import Config
import random

# Set seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Generates TF-IDF embeddings for ticket summaries and interaction content.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        np.ndarray: TF-IDF embeddings.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        min_df=4, 
        max_df=0.90
    )
    
    # Combine ticket summaries and interaction content
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    
    # Fit and transform data
    X = vectorizer.fit_transform(data).toarray()
    
    return X

def combine_embeddings(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Combines two sets of embeddings.
    
    Args:
        X1 (np.ndarray): First set of embeddings.
        X2 (np.ndarray): Second set of embeddings.
    
    Returns:
        np.ndarray: Combined embeddings.
    """
    return np.concatenate((X1, X2), axis=1)

Key Differences
Import Statement for Config:

Original: from Config import *
Improved: from Config import Config
Reason: Importing specific classes is preferable to avoid potential namespace conflicts.
Function Naming:

Original: get_tfidf_embd, combine_embd
Improved: get_tfidf_embeddings, combine_embeddings
Reason: Clearer and more descriptive function names improve readability.
Type Annotations:

Improved: Added type annotations for function parameters and return types.
Reason: Type annotations enhance code clarity and help with static analysis tools.
Docstrings:

Improved: Added docstrings to both functions.
Reason: Provides inline documentation, making the code easier to understand and maintain.
Code Comments:

Improved: Added a comment for setting the seed.
Reason: Comments clarify the purpose of code sections, particularly for reproducibility of results.
Overall, the improved code enhances readability, maintainability, and provides better documentation while maintaining the original functionality.

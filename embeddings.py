import numpy as np
import pandas as pd
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Generate enhanced TF-IDF embeddings by combining Ticket Summary and Interaction Content.

    Args:
        df (pd.DataFrame): DataFrame containing text fields.

    Returns:
        np.ndarray: TF-IDF embedding matrix.
    """
    # Basic preprocessing - simple version without NLTK dependencies
    df['processed_summary'] = df[Config.TICKET_SUMMARY].fillna('').str.lower()
    df['processed_content'] = df[Config.INTERACTION_CONTENT].fillna('').str.lower()

    # Create combined text
    combined_text = df['processed_summary'] + " " + df['processed_content']

    # Create the vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        max_features=1500,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    # Transform the text
    X = vectorizer.fit_transform(combined_text).toarray()

    logging.info(f"TF-IDF embeddings shape: {X.shape}")
    return X


def combine_embd(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Concatenate two embedding matrices.

    Args:
        X1 (np.ndarray): First embedding matrix.
        X2 (np.ndarray): Second embedding matrix.

    Returns:
        np.ndarray: Combined embedding matrix.
    """
    return np.concatenate((X1, X2), axis=1)
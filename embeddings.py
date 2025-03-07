import numpy as np
import pandas as pd
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO)


def get_tfidf_embd(df: pd.DataFrame) -> np.ndarray:
    """
    Generate TF-IDF embeddings by combining Ticket Summary and Interaction Content.

    Args:
        df (pd.DataFrame): DataFrame containing text fields.

    Returns:
        np.ndarray: TF-IDF embedding matrix.
    """
    text_data = df[Config.TICKET_SUMMARY] + " " + df[Config.INTERACTION_CONTENT]
    vectorizer = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    X = vectorizer.fit_transform(text_data).toarray()
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

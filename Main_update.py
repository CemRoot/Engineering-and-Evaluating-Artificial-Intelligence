from preprocess import get_input_data, de_duplication, noise_remover
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.chained_model import chained_model_predict
import numpy as np
from Config import Config

# Seed configuration for reproducibility
seed = 0
np.random.seed(seed)

def load_data() -> pd.DataFrame:
    """
    Loads input data.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    return get_input_data()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses data by removing duplicates and noise.
    
    Args:
        df (pd.DataFrame): Input data.
    
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings_and_df(df: pd.DataFrame) -> tuple:
    """
    Gets TF-IDF embeddings and associated DataFrame.
    
    Args:
        df (pd.DataFrame): Preprocessed data.
    
    Returns:
        tuple: Embeddings and DataFrame.
    """
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    """
    Creates data object with embeddings and DataFrame.
    
    Args:
        X (np.ndarray): Embeddings.
        df (pd.DataFrame): DataFrame.
    
    Returns:
        Data: Data object.
    """
    return Data(X, df)

if __name__ == '__main__':
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Ensure data types
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype('U')
    
    # Group data and process each group
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"\nProcessing group: {name}")
        X, group_df = get_embeddings_and_df(group_df)
        data = get_data_object(X, group_df)
        chained_model_predict(data, group_df, name)

Key Differences:
Imports:

The original code uses wildcard imports (*), while the updated code uses specific imports. This improves readability and helps avoid potential import conflicts.
Random Seed:

The random module seed setting is removed in the updated code, as it's not used in further operations.
Function Annotations:

The updated code includes type annotations and docstrings for better understanding and maintainability.
Data Type Conversion:

The use of .values.astype('U') is replaced with .astype('U') for direct DataFrame column type conversion. This is a more concise and modern approach.
Formatting:

Improved formatting and added comments for better readability and understanding.
Overall, the updated code is more concise, readable, and maintainable with better import practices, type annotations, and comments.

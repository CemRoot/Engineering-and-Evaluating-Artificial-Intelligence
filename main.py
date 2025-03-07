import random
import numpy as np
import pandas as pd
import argparse
import os
from preprocess import get_input_data, de_duplication, noise_remover
from embeddings import get_tfidf_embd
from modelling.data_model import Data
from modelling.chained_model import chained_model_predict
from modelling.hierarchical_model import hierarchical_model_predict
from Config import Config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add file handler separately to avoid permission issues
try:
    file_handler = logging.FileHandler("classification.log", mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info("Logging to file initialized")
except Exception as e:
    logging.warning(f"Could not create log file: {e}")

# Set random seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data() -> pd.DataFrame:
    """
    Loads and returns the input data from CSV files.
    """
    return get_input_data()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deduplication and noise removal to the input DataFrame.
    """
    df = de_duplication(df)
    df = noise_remover(df)
    return df


def get_embeddings_and_df(df: pd.DataFrame):
    """
    Generates TF-IDF embeddings and returns both the embeddings and the DataFrame.
    """
    X = get_tfidf_embd(df)
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame) -> Data:
    """
    Encapsulates embeddings and DataFrame into a Data object.
    """
    return Data(X, df)


if __name__ == '__main__':
    # Create directory for output artifacts
    os.makedirs("error_analysis", exist_ok=True)

    # Add command-line argument for model selection
    parser = argparse.ArgumentParser(description='Multi-label email classification')
    parser.add_argument('--model', type=str, choices=['chained', 'hierarchical', 'both'],
                        default='both', help='Modeling approach to use')
    args = parser.parse_args()

    try:
        # Load and preprocess data
        logging.info("Loading data...")
        df = load_data()
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        logging.info("Preprocessing data...")
        df = preprocess_data(df)
        logging.info(f"Data preprocessed: {df.shape[0]} rows remaining")

        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype('U')

        # Group data
        grouped_df = df.groupby(Config.GROUPED)

        # Track results for final comparison
        results = {
            'chained': {},
            'hierarchical': {}
        }

        # Process each group
        for name, group_df in grouped_df:
            logging.info(f"\nProcessing group: {name}")
            X, group_df = get_embeddings_and_df(group_df)
            data = get_data_object(X, group_df)

            # Run the selected modeling approach(es)
            if args.model in ['chained', 'both']:
                logging.info("\n==== Chained Multi-Output Classification ====")
                chained_model_predict(data, group_df, name)

            if args.model in ['hierarchical', 'both']:
                logging.info("\n==== Hierarchical Modeling ====")
                hierarchical_model_predict(data, group_df, name)

        logging.info("\nClassification completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
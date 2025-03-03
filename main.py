from preprocess import *
from embeddings import *
from modelling.data_model import *
from modelling.chained_model import chained_model_predict
import random
import numpy as np
from Config import Config

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    return get_input_data()

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings_and_df(df: pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print("\nProcessing group:", name)
        X, group_df = get_embeddings_and_df(group_df)
        data = get_data_object(X, group_df)
        chained_model_predict(data, group_df, name)

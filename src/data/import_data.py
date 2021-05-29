""" Function to import parallel sentences in two languages.
"""

import pandas as pd
import pickle
from src.data.preprocess_data import timer

@timer
def load_doc(filename):
    """ Function to load doc into memory. """
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text

@timer
def to_sentences(doc):
    """ Function to split a loaded document into sentences. """
    return doc.strip().split('\n')

@timer
def create_data_subset(sentence_data_source='../data/external/europarl-v7.de-en.en',
                       sentences_data_target='../data/external/europarl-v7.de-en.de',
                       sample_size=200000,
                       sentence_data_sampled_path = "../data/interim/europarl_english_german.pkl"):
    """ Function to import the data and concatenate it into a dataframe.

        Args:
            sentence_data_source (numpy.array): Array containing text
            sentences_data_target (numpy.array): Array containing text
            number_datapoints (int): Size of subset

        Returns:
            dataframe with source and target language sentences

        """
    doc_source = load_doc(sentence_data_source)
    sentences_source = to_sentences(doc_source)
    doc_target = load_doc(sentences_data_target)
    sentences_target = to_sentences(doc_target)

    df = pd.DataFrame({'text_source': sentences_source, 'text_target': sentences_target}, columns=['text_source',
                                                                                                   'text_target'])
    df_sampled = df.sample(sample_size, random_state=42).reset_index(drop=True).reset_index().rename(columns={"index": "id"})
    df_sampled.to_pickle(sentence_data_sampled_path)
    print("Sampled dataframe saved in: " + sentence_data_sampled_path)

@timer
def import_data(df_sampled_path="../data/interim/europarl_english_german.pkl"):
    """ Function to import the data and concatenate it into a dataframe.

    Args:
        sentence_data_source (numpy.array): Array containing text

    Returns:
        dataframe with source and target language sentences

    """
    with open(df_sampled_path, "rb") as input_file:
        return pickle.load(input_file)

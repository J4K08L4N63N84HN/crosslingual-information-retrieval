""" Functions to import europarl dataset and create random subset for our task.
"""

import pickle5 as pickle

import numpy as np
import pandas as pd

from src.utils.timer import timer


@timer
def load_doc(filename):
    """ Function to load doc into memory.
    """
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


@timer
def to_sentences(doc):
    """ Function to split a loaded document into sentences.
    """
    return doc.strip().split('\n')


@timer
def create_data_subset(sentence_data_source_path='../data/external/europarl-v7.de-en.en',
                       sentence_data_target_path='../data/external/europarl-v7.de-en.de',
                       sample_size=25000,
                       sentence_data_sampled_path="../data/interim/europarl_english_german.pkl"):
    """ Function to import an europarl dataset and save a subset as pickle.

        Args:
            sentence_data_source_path (string): Path to europarl data from source language.
            sentence_data_target_path (string): Path to europarl data from target language.
            sample_size (int): Size of sample subset.
            sentence_data_sampled_path (string): Path where to save pickle containing sampled parallel data.
        """
    doc_source = load_doc(sentence_data_source_path)
    sentences_source = to_sentences(doc_source)
    doc_target = load_doc(sentence_data_target_path)
    sentences_target = to_sentences(doc_target)

    df = pd.DataFrame({'text_source': sentences_source, 'text_target': sentences_target}, columns=['text_source',
                                                                                                   'text_target'])
    # drop empty sentences
    not_empty_sentence_index = np.logical_not(np.logical_or(df.text_source == "", df.text_target == ""))
    df_not_empty = df.loc[not_empty_sentence_index]

    df_sampled = df_not_empty.sample(sample_size, random_state=42).reset_index(drop=True).reset_index().rename(
        columns={"index": "id_source"})
    df_sampled["id_target"] = df_sampled["id_source"]
    df_sampled.to_pickle(sentence_data_sampled_path)
    print("Sampled dataframe saved in: " + sentence_data_sampled_path)


@timer
def import_data(df_sampled_path="../data/interim/europarl_english_german.pkl"):
    """ Function to import data.
    """
    with open(df_sampled_path, "rb") as input_file:
        return pickle.load(input_file)

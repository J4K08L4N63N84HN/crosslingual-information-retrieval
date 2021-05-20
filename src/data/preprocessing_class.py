""" Class for creating a parallel sentence dataset.
"""
import string

import numpy as np
import pandas as pd

from src.data.import_data import import_data
from src.data.preprocess_data import create_cleaned_token, create_cleaned_text, number_punctuations_total, number_words, \
    number_unique_words, number_punctuation_marks, number_characters, number_pos, number_times, polarity, subjectivity, \
    number_stopwords, named_entities, word_embeddings, remove_stopwords, average_characters



class PreprocessingEuroParl:
    """ Class for preprocessing the EuroParl datasets for two languages.

    Attributes:
        dataframe (dataframe): Parallel sentences of the europarl dataset
        punctuation_list (list): Punctuation list for removal and counting
        pos_list (list): Part of speech tags list for counting
        tense_list (list): Tense list for counting
        preprocessed_source (dataframe): Preprocessed dataframe for source sentences
        preprocessed_target (dataframe): Preprocessed dataframe for target sentences
        preprocessed_dataframe (dataframe): Merged dataframe from source and target
    """

    def __init__(self, sentence_data_source='../../data/external/europarl-v7.de-en.en',
                 sentence_data_target='../../data/external/europarl-v7.de-en.de',
                 number_datapoints=100):
        """ Initialize dataframe by importing europarl data for source and target

        Args:
            sentence_data_source (str): Path of the europarl source dataset
            sentence_data_target (str): Path of the europarl target dataset
            number_datapoints (int): Size of the sample of the europarl dataset
        """
        self.dataframe = import_data(sentence_data_source, sentence_data_target, number_datapoints)
        self.punctuation_list = list(string.punctuation)
        self.pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRT', 'PRON',
                         'PROPN', 'SCONJ', 'SYM', 'VERB', 'X']
        self.tense_list = ['Pres', 'Past', '']
        self.preprocessed_source = pd.DataFrame()
        self.preprocessed_target = pd.DataFrame()
        self.preprocessed_dataframe = pd.DataFrame()

    def preprocess_sentences_source(self, stopwords_source, nlp_source, textblob_source, embedding_matrix_source,
                                    embedding_dictionary_source):
        """ Preprocess the source sentence dataset

        Args:
            stopwords_source (list): List of stopwords to remove and count
            nlp_source (spacy pipeline): Spacy pipeline for preprocessing
            textblob_source (textblob object): Textblob object for sentiment analysis
            embedding_matrix_source (str): Path to embedding matrix
            embedding_dictionary_source (str): Path to embedding dictionary
        """
        self.dataframe["token_preprocessed_source"] = create_cleaned_token((self.dataframe["text_source"]),
                                                                           nlp_source, stopwords_source)
        self.dataframe["text_source_1"] = create_cleaned_text(self.dataframe["text_source"])
        # count stopwords before removing
        self.preprocessed_source["number_stopwords_source"] = number_stopwords(self.dataframe["text_source_1"],
                                                                               stopwords_source)
        self.dataframe["text_source_1"] = remove_stopwords(self.dataframe["text_source_1"], stopwords_source)
        self.preprocessed_source["number_punctuations_total_source"] = number_punctuations_total(
            self.dataframe["text_source_1"])
        self.preprocessed_source["number_words_source"] = number_words(self.dataframe["text_source_1"])
        self.preprocessed_source["number_unique_words_source"] = number_unique_words(self.dataframe["text_source_1"])
        self.preprocessed_source["number_characters_source"] = number_characters(self.dataframe["text_source_1"])
        self.preprocessed_source["characters_avg_source"] = average_characters(self.preprocessed_source["number_characters_source"],
                                                                               self.preprocessed_source["number_words_source"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed_source[f"number_{punctuation_mark}_source"] = number_punctuation_marks(self.dataframe[
                                                                                                         "text_source_1"],
                                                                                                     punctuation_mark)
        for pos in self.pos_list:
            self.preprocessed_source[f"number_{pos}_source"] = number_pos(self.dataframe["text_source"], nlp_source,
                                                                          pos)
        for tense in self.tense_list:
            self.preprocessed_source[f"number_{tense}_source"] = number_times(self.dataframe["text_source"],
                                                                              nlp_source,
                                                                              tense)
        self.preprocessed_source["score_polarity_source"] = polarity(self.dataframe["text_source"], textblob_source)
        self.preprocessed_source["score_subjectivity_source"] = subjectivity(self.dataframe["text_source"],
                                                                             textblob_source)
        self.preprocessed_source["list_named_entities_source"] = named_entities(self.dataframe["text_source"],
                                                                                nlp_source)
        self.preprocessed_source["sentence_embedding_source"] = word_embeddings(
            self.dataframe["token_preprocessed_source"],
            embedding_matrix_source,
            embedding_dictionary_source)
        self.preprocessed_source["score_polarity_source"] = polarity(self.dataframe["text_source"], textblob_source)
        self.preprocessed_source["score_subjectivity_source"] = subjectivity(self.dataframe["text_source"],
                                                                             textblob_source)
        self.preprocessed_source["number_stopwords_source"] = number_stopwords(self.dataframe["text_source"],
                                                                               stopwords_source)
        self.preprocessed_source["list_named_entities_source"] = named_entities(self.dataframe["text_source"],
                                                                                nlp_source)
        self.preprocessed_source["sentence_embedding_source"] = word_embeddings(
            self.dataframe["token_preprocessed_source"],
            embedding_matrix_source,
            embedding_dictionary_source)

    def preprocess_sentences_target(self, stopwords_target, nlp_target, textblob_target, embedding_matrix_target,
                                    embedding_dictionary_target):
        """ Preprocess the target sentence dataset

        Args:
            stopwords_target (list): List of stopwords to remove and count
            nlp_target (spacy pipeline): Spacy pipeline for preprocessing
            textblob_target (textblob object): Textblob object for sentiment analysis
            embedding_matrix_target (str): Path to embedding matrix
            embedding_dictionary_target (str): Path to embedding dictionary
        """
        self.dataframe["token_preprocessed_target"] = create_cleaned_token((self.dataframe["text_target"]),
                                                                           nlp_target, stopwords_target)
        self.dataframe["text_target_1"] = create_cleaned_text(self.dataframe["text_target"])
        self.preprocessed_target["number_stopwords_target"] = number_stopwords(self.dataframe["text_target_1"],
                                                                               stopwords_target)
        self.dataframe["text_target_1"] = remove_stopwords(self.dataframe["text_target_1"], stopwords_target)
        self.preprocessed_target["number_punctuations_total_target"] = number_punctuations_total(
            self.dataframe["text_target_1"])
        self.preprocessed_target["number_words_target"] = number_words(self.dataframe["text_target_1"])
        self.preprocessed_target["number_unique_words_target"] = number_unique_words(self.dataframe["text_target_1"])
        self.preprocessed_target["number_characters_target"] = number_characters(self.dataframe["text_target_1"])
        self.preprocessed_target["characters_avg_target"] = average_characters(self.preprocessed_target["number_characters_target"],
                                                                               self.preprocessed_target["number_words_target"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed_target[f"number_{punctuation_mark}_target"] = number_punctuation_marks(self.dataframe[
                                                                                                         "text_target_1"],
                                                                                                     punctuation_mark)
        for pos in self.pos_list:
            self.preprocessed_target[f"number_{pos}_target"] = number_pos(self.dataframe["text_target"], nlp_target,
                                                                          pos)
        for tense in self.tense_list:
            self.preprocessed_target[f"number_{tense}_target"] = number_times(self.dataframe["text_target"],
                                                                              nlp_target,
                                                                              tense)
        self.preprocessed_target["score_polarity_target"] = polarity(self.dataframe["text_target"], textblob_target)
        self.preprocessed_target["score_subjectivity_target"] = subjectivity(self.dataframe["text_target"],
                                                                             textblob_target)
        self.preprocessed_target["list_named_entities_target"] = named_entities(self.dataframe["text_target"],
                                                                                nlp_target)
        self.preprocessed_target["sentence_embedding_target"] = word_embeddings(
            self.dataframe["token_preprocessed_target"],
            embedding_matrix_target,
            embedding_dictionary_target)
        self.preprocessed_target["score_polarity_target"] = polarity(self.dataframe["text_target"], textblob_target)
        self.preprocessed_target["score_subjectivity_target"] = subjectivity(self.dataframe["text_target"],
                                                                             textblob_target)
        self.preprocessed_target["number_stopwords_target"] = number_stopwords(self.dataframe["text_target"],
                                                                               stopwords_target)
        self.preprocessed_target["list_named_entities_target"] = named_entities(self.dataframe["text_target"],
                                                                                nlp_target)
        self.preprocessed_target["sentence_embedding_target"] = word_embeddings(
            self.dataframe["token_preprocessed_target"],
            embedding_matrix_target, embedding_dictionary_target)

    def combine_source_target(self):
        """ Combine preprocessed source and target sentences in one dataframe
        """
        self.preprocessed_dataframe = pd.concat([self.preprocessed_source, self.preprocessed_target], axis=1)

    def add_label(self):
        """ Add translation label to the dataframe
        """
        self.preprocessed_dataframe["Translation"] = np.ones((int(self.preprocessed_dataframe.shape[0]), 1),
                                                             dtype=np.int8)

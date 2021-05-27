""" Class for creating a parallel sentence dataset.
"""
import string

import numpy as np
import pandas as pd

from src.data.import_data import import_data
from src.data.preprocess_data import create_cleaned_text, number_punctuations_total, number_words, \
    number_unique_words, number_punctuation_marks, number_characters, number_pos, number_times, polarity, subjectivity, \
    number_stopwords, named_entities, word_embeddings, remove_stopwords, average_characters, load_embeddings, \
    translate_words, create_cleaned_token_embedding, tf_idf_vector, sentence_embedding_average, \
    sentence_embedding_tf_idf


class PreprocessingEuroParl:
    """ Class for preprocessing the EuroParl datasets for two languages.

    Attributes:
        dataframe (dataframe): Parallel sentences of the europarl dataset
        punctuation_list (list): Punctuation list for removal and counting
        pos_list (list): Part of speech tags list for counting
        tense_list (list): Tense list for counting
        preprocessed (dataframe): Preprocessed dataframe for source sentences
        preprocessed (dataframe): Preprocessed dataframe for target sentences
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
        self.preprocessed = pd.DataFrame()
        self.preprocessed = pd.DataFrame()
        self.preprocessed_dataframe = pd.DataFrame()

    def preprocess_sentences(self, stopwords_source, nlp_source, textblob_source, embedding_matrix_source_path,
                             embedding_dictionary_source_path, stopwords_target, nlp_target, textblob_target,
                             embedding_matrix_target_path,
                             embedding_dictionary_target_path,
                             n_neighbors):
        """ Preprocess the source sentence dataset

        Args:
            stopwords_source (list): List of stopwords to remove and count
            nlp_source (spacy pipeline): Spacy pipeline for preprocessing
            textblob_source (textblob object): Textblob object for sentiment analysis
            embedding_matrix_source_path (str): Path to embedding matrix
            embedding_dictionary_source_path (str): Path to embedding dictionary
            stopwords_target (list): List of stopwords to remove and count
            nlp_target (spacy pipeline): Spacy pipeline for preprocessing
            textblob_target (textblob object): Textblob object for sentiment analysis
            embedding_matrix_target_path (str): Path to embedding matrix
            embedding_dictionary_target_path (str): Path to embedding dictionary
        """
        self.dataframe["token_preprocessed_embedding_source"] = create_cleaned_token_embedding((self.dataframe[
            "text_source"]),
                                                                                               nlp_source,
                                                                                               stopwords_source)
        self.dataframe["text_preprocessed_source"] = create_cleaned_text(self.dataframe["text_source"])
        # count stopwords before removing
        self.preprocessed["number_stopwords_source"] = number_stopwords(self.dataframe["text_preprocessed_source"],
                                                                        stopwords_source)
        self.dataframe["text_preprocessed_source"] = remove_stopwords(self.dataframe["text_preprocessed_source"],
                                                                      stopwords_source)
        self.preprocessed["number_punctuations_total_source"] = number_punctuations_total(
            self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_words_source"] = number_words(self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_unique_words_source"] = number_unique_words(
            self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_characters_source"] = number_characters(self.dataframe["text_preprocessed_source"])
        self.preprocessed["characters_avg_source"] = average_characters(
            self.preprocessed["number_characters_source"],
            self.preprocessed["number_words_source"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed[f"number_{punctuation_mark}_source"] = number_punctuation_marks(self.dataframe[
                                                                                                  "text_preprocessed_source"],
                                                                                              punctuation_mark)
        for pos in self.pos_list:
            self.preprocessed[f"number_{pos}_source"] = number_pos(self.dataframe["text_source"], nlp_source,
                                                                   pos)
        for tense in self.tense_list:
            self.preprocessed[f"number_{tense}_source"] = number_times(self.dataframe["text_source"],
                                                                       nlp_source,
                                                                       tense)
        self.preprocessed["score_polarity_source"] = polarity(self.dataframe["text_source"], textblob_source)
        self.preprocessed["score_subjectivity_source"] = subjectivity(self.dataframe["text_source"],
                                                                      textblob_source)
        self.preprocessed["list_named_entities_source"] = named_entities(self.dataframe["text_source"],
                                                                         nlp_source)
        self.preprocessed["score_polarity_source"] = polarity(self.dataframe["text_source"], textblob_source)
        self.preprocessed["score_subjectivity_source"] = subjectivity(self.dataframe["text_source"],
                                                                      textblob_source)
        self.preprocessed["number_stopwords_source"] = number_stopwords(self.dataframe["text_source"],
                                                                        stopwords_source)
        self.preprocessed["list_named_entities_source"] = named_entities(self.dataframe["text_source"],
                                                                         nlp_source)


        # Preprocess target data
        self.dataframe["token_preprocessed_embedding_target"] = create_cleaned_token_embedding((self.dataframe[
            "text_target"]),
                                                                                               nlp_target,
                                                                                               stopwords_target)
        self.dataframe["text_preprocessed_target"] = create_cleaned_text(self.dataframe["text_target"])
        self.preprocessed["number_stopwords_target"] = number_stopwords(self.dataframe["text_preprocessed_target"],
                                                                        stopwords_target)
        self.dataframe["text_preprocessed_target"] = remove_stopwords(self.dataframe["text_preprocessed_target"],
                                                                      stopwords_target)
        self.preprocessed["number_punctuations_total_target"] = number_punctuations_total(
            self.dataframe["text_preprocessed_target"])
        self.preprocessed["number_words_target"] = number_words(self.dataframe["text_preprocessed_target"])
        self.preprocessed["number_unique_words_target"] = number_unique_words(
            self.dataframe["text_preprocessed_target"])
        self.preprocessed["number_characters_target"] = number_characters(self.dataframe["text_preprocessed_target"])
        self.preprocessed["characters_avg_target"] = average_characters(
            self.preprocessed["number_characters_target"],
            self.preprocessed["number_words_target"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed[f"number_{punctuation_mark}_target"] = number_punctuation_marks(self.dataframe[
                                                                                                  "text_preprocessed_target"],
                                                                                              punctuation_mark)
        for pos in self.pos_list:
            self.preprocessed[f"number_{pos}_target"] = number_pos(self.dataframe["text_target"], nlp_target,
                                                                   pos)
        for tense in self.tense_list:
            self.preprocessed[f"number_{tense}_target"] = number_times(self.dataframe["text_target"],
                                                                       nlp_target,
                                                                       tense)
        self.preprocessed["score_polarity_target"] = polarity(self.dataframe["text_target"], textblob_target)
        self.preprocessed["score_subjectivity_target"] = subjectivity(self.dataframe["text_target"],
                                                                      textblob_target)
        self.preprocessed["list_named_entities_target"] = named_entities(self.dataframe["text_target"],
                                                                         nlp_target)
        self.preprocessed["score_polarity_target"] = polarity(self.dataframe["text_target"], textblob_target)
        self.preprocessed["score_subjectivity_target"] = subjectivity(self.dataframe["text_target"],
                                                                      textblob_target)
        self.preprocessed["number_stopwords_target"] = number_stopwords(self.dataframe["text_target"],
                                                                        stopwords_target)
        self.preprocessed["list_named_entities_target"] = named_entities(self.dataframe["text_target"],
                                                                         nlp_target)

        embedding_matrix_normalized_source, embedding_dictionary_source = load_embeddings(
            embedding_matrix_source_path, embedding_dictionary_source_path)
        embedding_matrix_normalized_target, embedding_dictionary_target = load_embeddings(
            embedding_matrix_target_path, embedding_dictionary_target_path)

        self.preprocessed["word_embedding_source"] = word_embeddings(
            self.dataframe["token_preprocessed_embedding_source"],
            embedding_matrix_normalized_source,
            embedding_dictionary_source)
        self.preprocessed["word_embedding_target"] = word_embeddings(
            self.dataframe["token_preprocessed_embedding_target"],
            embedding_matrix_normalized_target,
            embedding_dictionary_target)

        self.preprocessed["translated_to_target_source"] = translate_words(
            self.dataframe["token_preprocessed_embedding_source"],
            embedding_dictionary_source,
            embedding_matrix_normalized_target,
            embedding_dictionary_target,
            embedding_matrix_normalized_target,
            n_neighbors)
        self.preprocessed["translated_to_source_target"] = translate_words(
            self.dataframe["token_preprocessed_embedding_target"],
            embedding_dictionary_source,
            embedding_matrix_normalized_target,
            embedding_dictionary_target,
            embedding_matrix_normalized_target,
            n_neighbors)

        self.preprocessed["tf_idf_source"] = tf_idf_vector(self.dataframe["token_preprocessed_embedding_source"])
        self.preprocessed["tf_idf_target"] = tf_idf_vector(self.dataframe["token_preprocessed_embedding_target"])

        self.preprocessed["sentence_embedding_average_source"] = sentence_embedding_average(
            self.preprocessed["word_embedding_source"])
        self.preprocessed["sentence_embedding_average_target"] = sentence_embedding_average(
            self.preprocessed["word_embedding_target"])

        self.preprocessed["sentence_embedding_tf_idf_source"] = sentence_embedding_tf_idf(
            self.preprocessed["word_embedding_source"],
            self.preprocessed["tf_idf_source"])
        self.preprocessed["sentence_embedding_tf_idf_target"] = sentence_embedding_tf_idf(
            self.preprocessed["word_embedding_target"],
            self.preprocessed["tf_idf_target"])

        self.preprocessed["Translation"] = np.ones((int(self.preprocessed.shape[0]), 1),
                                                   dtype=np.int8)
        self.preprocessed.reset_index(inplace = True, drop = True)
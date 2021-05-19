""" Class for creating a parallel sentence dataset.
"""
import string

import numpy as np
import pandas as pd

from src.data.import_data import import_data
from src.data.preprocess_data import create_cleaned_token, create_cleaned_text, number_punctuations_total, number_words, \
    number_unique_words, number_punctuation_marks, number_characters, number_pos, number_times, polarity, subjectivity, \
    number_stopwords, named_entities, word_embeddings, remove_stopwords

class PreprocessingEuroParl:
    def __init__(self, sentence_data_source='../../data/external/europarl-v7.de-en.en',
                 sentence_data_target='../../data/external/europarl-v7.de-en.de',
                 number_datapoints=100):
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
        self.dataframe["token_preprocessed_source"] = create_cleaned_token((self.dataframe["text_source"]),
                                                                           nlp_source, stopwords_source)
        self.dataframe["text_source_1"] = create_cleaned_text(self.dataframe["text_source"])
        # count stopwords before removing
        self.preprocessed_source["number_stopwords_source"] = number_stopwords(self.dataframe["text_source_1"],
                                                                     stopwords_source)
        self.dataframe["text_source_1"] = remove_stopwords(self.dataframe["text_source_1"], stopwords_source)
        self.preprocessed_source["number_punctuations_total_source"] = number_punctuations_total(self.dataframe["text_source_1"])
        self.preprocessed_source["number_words_source"] = number_words(self.dataframe["text_source_1"])
        self.preprocessed_source["number_unique_words_source"] = number_unique_words(self.dataframe["text_source_1"])
        self.preprocessed_source["number_characters_source"] = number_characters(self.dataframe["text_source_1"])
        self.preprocessed_source["number_punctuations_total_source"] = number_punctuations_total(
            self.dataframe["text_source"])
        self.preprocessed_source["number_words_source"] = number_words(self.dataframe["text_source"])
        self.preprocessed_source["number_unique_words_source"] = number_unique_words(self.dataframe["text_source"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed_source[f"number_{punctuation_mark}_source"] = number_punctuation_marks(self.dataframe[
                                                                                               "text_source_1"],
                                                                                           punctuation_mark)
        self.preprocessed_source["number_characters_source"] = number_characters(self.dataframe["text_source"])
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
        self.preprocessed_source["list_named_entities_source"] = named_entities(self.dataframe["text_source"], nlp_source)
        self.preprocessed_source["sentence_embedding_source"] = word_embeddings(self.dataframe["token_preprocessed_source"],
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
        self.dataframe["token_preprocessed_target"] = create_cleaned_token((self.dataframe["text_target"]),
                                                                           nlp_target, stopwords_target)
        self.dataframe["text_target_1"] = create_cleaned_text(self.dataframe["text_target"])
        self.dataframe["number_stopwords_target"] = number_stopwords(self.dataframe["text_target_1"],
                                                                     stopwords_target)
        self.dataframe["text_target_1"] = remove_stopwords(self.dataframe["text_target_1"], stopwords_target)
        self.preprocessed_target["number_punctuations_total_target"] = number_punctuations_total(self.dataframe["text_target_1"])
        self.preprocessed_target["number_words_target"] = number_words(self.dataframe["text_target_1"])
        self.preprocessed_target["number_unique_words_target"] = number_unique_words(self.dataframe["text_target_1"])
        self.preprocessed_target["number_characters_target"] = number_characters(self.dataframe["text_target_1"])
        self.preprocessed_target["number_punctuations_total_target"] = number_punctuations_total(
            self.dataframe["text_target"])
        self.preprocessed_target["number_words_target"] = number_words(self.dataframe["text_target"])
        self.preprocessed_target["number_unique_words_target"] = number_unique_words(self.dataframe["text_target"])
        for punctuation_mark in self.punctuation_list:
            self.preprocessed_target[f"number_{punctuation_mark}_target"] = number_punctuation_marks(self.dataframe[
                                                                                               "text_target_1"],
                                                                                           punctuation_mark)
        self.preprocessed_target["number_characters_target"] = number_characters(self.dataframe["text_target"])
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
        self.preprocessed_target["list_named_entities_target"] = named_entities(self.dataframe["text_target"], nlp_target)
        self.preprocessed_target["sentence_embedding_target"] = word_embeddings(self.dataframe["token_preprocessed_target"],
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
        self.preprocessed_dataframe = pd.concat([self.preprocessed_source, self.preprocessed_target], axis=1)

    def add_label(self):
        self.preprocessed_dataframe["Translation"] = np.ones((int(self.preprocessed_dataframe.shape[0]), 1),
                                                             dtype=np.int8)

""" Class for creating a parallel sentence dataset.
"""
import string

import numpy as np
import pandas as pd

from src.data.import_data import import_data
from src.data.preprocess_data import create_cleaned_text, number_punctuations_total, number_words, \
    number_unique_words, number_punctuation_marks, number_characters, word_embeddings, \
    average_characters, load_embeddings, \
    translate_words, create_cleaned_token_embedding, tf_idf_vector, sentence_embedding_average, \
    sentence_embedding_tf_idf, named_numbers, create_translation_dictionary, number_pos, number_times, \
    spacy


class PreprocessingEuroParl:
    """ Class for preprocessing the EuroParl datasets for two languages.

    Attributes:
        dataframe (dataframe): Parallel sentences of the europarl dataset
        punctuation_list (list): Punctuation list for removal and counting
        pos_list (list): Part of speech tags list for counting
        tense_list (list): Tense list for counting
        embedding_list (list): List of embeddings
        preprocessed (dataframe): Preprocessed dataframe for source sentences
    """

    def __init__(self, df_sampled_path="../data/interim/europarl_english_german.pkl"):
        """ Initialize dataframe by using saved data.

            Args:
                df_sampled_path (str): Path to saved dataset.
        """
        self.dataframe = import_data(df_sampled_path)
        self.punctuation_list = list(string.punctuation)
        self.pos_list = ['ADJ', 'NOUN', 'VERB']
        self.tense_list = ['Pres', 'Past', '']
        self.embedding_list = []
        self.preprocessed = pd.DataFrame()

    def preprocess_sentences(self, nlp_source, nlp_target, stopword_source, stopword_target):
        """ Preprocess the source sentence dataset

            Args:
                stopwords_source (list): List of stopwords to remove and count.
                stopwords_target (list): List of stopwords to remove and count.
                nlp_source (spacy pipeline): Spacy pipeline for preprocessing.
                nlp_target (spacy pipeline): Spacy pipeline for preprocessing.

        """
        self.preprocessed["id_source"] = self.dataframe["id_source"]
        self.preprocessed["id_target"] = self.dataframe["id_target"]
        self.preprocessed["token_preprocessed_embedding_source"] = create_cleaned_token_embedding((self.dataframe[
            "text_source"]), nlp_source, stopword_source)
        self.preprocessed["token_preprocessed_embedding_target"] = create_cleaned_token_embedding((self.dataframe[
            "text_target"]), nlp_target, stopword_target)

        self.dataframe["text_preprocessed_source"] = create_cleaned_text(self.dataframe["text_source"], stopword_source)
        self.dataframe["text_preprocessed_target"] = create_cleaned_text(self.dataframe["text_target"], stopword_target)

        self.preprocessed["Translation"] = np.ones((int(self.preprocessed.shape[0]), 1),
                                                   dtype=np.int8)
        self.preprocessed.reset_index(inplace=True, drop=True)

    def extract_sentence_information(self, nlp_source, nlp_target):
        """

        Args:
            nlp_source (spacy pipeline): Spacy pipeline for preprocessing.
            nlp_target (spacy pipeline): Spacy pipeline for preprocessing.
            textblob_source (textblob object): Textblob object for sentiment analysis.
            textblob_target (textblob object): Textblob object for sentiment analysis.
        """
        self.preprocessed["number_punctuations_total_source"] = number_punctuations_total(
            self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_punctuations_total_target"] = number_punctuations_total(
            self.dataframe["text_preprocessed_target"])

        self.preprocessed["number_words_source"] = number_words(self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_words_target"] = number_words(self.dataframe["text_preprocessed_target"])

        self.preprocessed["number_unique_words_source"] = number_unique_words(
            self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_unique_words_target"] = number_unique_words(
            self.dataframe["text_preprocessed_target"])

        self.preprocessed["number_characters_source"] = number_characters(self.dataframe["text_preprocessed_source"])
        self.preprocessed["number_characters_target"] = number_characters(self.dataframe["text_preprocessed_target"])

        self.preprocessed["characters_avg_source"] = average_characters(
            self.preprocessed["number_characters_source"],
            self.preprocessed["number_words_source"])
        self.preprocessed["characters_avg_target"] = average_characters(
            self.preprocessed["number_characters_target"],
            self.preprocessed["number_words_target"])

        for punctuation_mark in self.punctuation_list:
            self.preprocessed[f"number_{punctuation_mark}_source"] = number_punctuation_marks(self.dataframe[
                                                                                                  "text_preprocessed_source"],
                                                                                              punctuation_mark)
            self.preprocessed[f"number_{punctuation_mark}_target"] = number_punctuation_marks(self.dataframe[
                                                                                                  "text_preprocessed_target"],
                                                                                              punctuation_mark)

        self.dataframe["text_source_spacy"] = spacy(self.dataframe["text_source"], nlp_source)
        self.dataframe["text_target_spacy"] = spacy(self.dataframe["text_target"], nlp_target)

        for pos in self.pos_list:
            self.preprocessed[f"number_{pos}_source"] = number_pos(self.dataframe["text_source_spacy"], nlp_source,
                                                                   pos)
            self.preprocessed[f"number_{pos}_target"] = number_pos(self.dataframe["text_target_spacy"], nlp_target,
                                                                   pos)

        for tense in self.tense_list:
            self.preprocessed[f"number_{tense}_source"] = number_times(self.dataframe["text_source_spacy"],
                                                                       nlp_source,
                                                                       tense)
            self.preprocessed[f"number_{tense}_target"] = number_times(self.dataframe["text_target_spacy"],
                                                                       nlp_target,
                                                                       tense)

        self.preprocessed["list_named_numbers_source"] = named_numbers(self.dataframe["text_source"])
        self.preprocessed["list_named_numbers_target"] = named_numbers(self.dataframe["text_target"])

    def create_embedding_information(self, embedding):
        """ Create information based on embeddings.

            Args:
                embedding (str): Type of embedding to create information.

        """
        embedding_array_source_path = "../data/interim/en_de_" + embedding + "_src_emb.pkl"
        embedding_dictionary_source_path = "../data/interim/en_de_" + embedding + "_src_word.pkl"
        embedding_array_target_path = "../data/interim/en_de_" + embedding + "_trg_emb.pkl"
        embedding_dictionary_target_path = "../data/interim/en_de_" + embedding + "_trg_word.pkl"

        embedding_array_normalized_source, embedding_dictionary_source = load_embeddings(
            embedding_array_source_path, embedding_dictionary_source_path)
        embedding_array_normalized_target, embedding_dictionary_target = load_embeddings(
            embedding_array_target_path, embedding_dictionary_target_path)

        self.dataframe[f"word_embedding_{embedding}_source"] = word_embeddings(
            self.preprocessed["token_preprocessed_embedding_source"],
            embedding_array_normalized_source,
            embedding_dictionary_source)
        self.dataframe[f"word_embedding_{embedding}_target"] = word_embeddings(
            self.preprocessed["token_preprocessed_embedding_target"],
            embedding_array_normalized_target,
            embedding_dictionary_target)

        translation_to_target_source, translation_to_source_target = create_translation_dictionary(
            self.preprocessed[
                "token_preprocessed_embedding_source"],
            self.preprocessed[
                "token_preprocessed_embedding_target"],
            embedding_array_normalized_source,
            embedding_dictionary_source,
            embedding_array_normalized_target,
            embedding_dictionary_target)

        self.preprocessed[f"translated_to_target_{embedding}_source"] = translate_words(
            self.preprocessed["token_preprocessed_embedding_source"],
            translation_to_target_source)
        self.preprocessed[f"translated_to_source_{embedding}_target"] = translate_words(
            self.preprocessed["token_preprocessed_embedding_target"],
            translation_to_source_target)

        self.dataframe[f"tf_idf_{embedding}_source"] = tf_idf_vector(self.preprocessed[
                                                                         "token_preprocessed_embedding_source"])
        self.dataframe[f"tf_idf_{embedding}_target"] = tf_idf_vector(self.preprocessed[
                                                                         "token_preprocessed_embedding_target"])

        self.preprocessed[f"sentence_embedding_average_{embedding}_source"] = sentence_embedding_average(
            self.dataframe[f"word_embedding_{embedding}_source"])
        self.preprocessed[f"sentence_embedding_average_{embedding}_target"] = sentence_embedding_average(
            self.dataframe[f"word_embedding_{embedding}_target"])

        self.preprocessed[f"sentence_embedding_tf_idf_{embedding}_source"] = sentence_embedding_tf_idf(
            self.dataframe[f"word_embedding_{embedding}_source"],
            self.dataframe[f"tf_idf_{embedding}_source"])
        self.preprocessed[f"sentence_embedding_tf_idf_{embedding}_target"] = sentence_embedding_tf_idf(
            self.dataframe[f"word_embedding_{embedding}_target"],
            self.dataframe[f"tf_idf_{embedding}_target"])

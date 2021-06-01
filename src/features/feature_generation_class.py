""" Class for creating features from preprocessed data.
"""

import pandas as pd

from src.features.embed_based import cosine_similarity_vector, jaccard, embedding_difference, euclidean_distance_vector, \
    word_mover_distance_vector
from src.features.sentence_based import difference_numerical, relative_difference_numerical, \
    normalized_difference_numerical


class FeatureGeneration:
    """ Class for generating features from preprocessed parallel sentences.

    Attributes:
        preprocessed_dataset (dataframe): Preprocessed dataset.
        feature_dataframe (dataframe): Dataset containing feature for training a model.
        feature_difference_list (list): List of preprocessed columns that should be compared.
        k (int): Number of principal components.
    """

    def __init__(self, dataset, k):
        """ Initialize dataframe by importing preprocessed dataset..

            Args:
                dataset (str): Path of the europarl source dataset.
                k (int): Number of principal components.
        """
        self.preprocessed_dataset = dataset
        self.feature_dataframe = pd.DataFrame()
        self.feature_difference_list = [
            'number_punctuations_total', 'number_words',
            'number_unique_words', 'number_!', 'number_"',
            'number_#', 'number_$', 'number_%',
            'number_&', "number_'", 'number_(',
            'number_)', 'number_*', 'number_+',
            'number_,', 'number_-', 'number_.',
            'number_/', 'number_:', 'number_;',
            'number_<', 'number_=', 'number_>',
            'number_?', 'number_@', 'number_[',
            "number_\\", "number_]", "number_^",
            'number__', 'number_`', 'number_{',
            'number_|', 'number_}', 'number_~',
            'number_characters', 'characters_avg',
            'number_ADJ',
            'number_NOUN', 'number_VERB',
            'number_Pres', 'number_Past', 'number_',
            'score_polarity', 'score_subjectivity',
            'number_stopwords']
        self.k = k

    def feature_generation(self):
        """ Create features of preprocessed columns.
        """
        self.feature_dataframe["source_id"] = self.preprocessed_dataset.id_source
        self.feature_dataframe["target_id"] = self.preprocessed_dataset.id_target

        for feature in self.feature_difference_list:
            self.feature_dataframe[f"{feature}_difference"] = difference_numerical(
                self.preprocessed_dataset[f"{feature}_source"], self.preprocessed_dataset[f"{feature}_target"])
            self.feature_dataframe[f"{feature}_difference_relative"] = relative_difference_numerical(
                self.preprocessed_dataset[f"{feature}_source"], self.preprocessed_dataset[f"{feature}_target"])
            self.feature_dataframe[f"{feature}_difference_normalized"] = normalized_difference_numerical(
                self.preprocessed_dataset[f"{feature}_source"], self.preprocessed_dataset[f"{feature}_target"],
                (self.preprocessed_dataset["number_punctuations_total_source"] + self.preprocessed_dataset[
                    "number_words_source"]),
                (self.preprocessed_dataset["number_punctuations_total_target"] + self.preprocessed_dataset[
                    "number_words_target"]))

        self.feature_dataframe["word_mover_distance"] = word_mover_distance_vector(
            self.preprocessed_dataset[
                "word_embedding_source"],
            self.preprocessed_dataset[
                "word_embedding_target"])

        self.feature_dataframe["cosine_similarity_average"] = cosine_similarity_vector(
            self.preprocessed_dataset[
                "sentence_embedding_average_source"],
            self.preprocessed_dataset[
                "sentence_embedding_average_target"])

        self.feature_dataframe["cosine_similarity_tf_idf"] = cosine_similarity_vector(
            self.preprocessed_dataset[
                "sentence_embedding_tf_idf_source"],
            self.preprocessed_dataset[
                "sentence_embedding_tf_idf_target"])

        self.feature_dataframe["euclidean_distance_average"] = euclidean_distance_vector(
            self.preprocessed_dataset[
                "sentence_embedding_average_source"],
            self.preprocessed_dataset[
                "sentence_embedding_average_target"])

        self.feature_dataframe["euclidean_distance_tf_idf"] = euclidean_distance_vector(
            self.preprocessed_dataset[
                "sentence_embedding_tf_idf_source"],
            self.preprocessed_dataset[
                "sentence_embedding_tf_idf_target"])

        self.feature_dataframe["jaccard_translation_source"] = jaccard(
            self.preprocessed_dataset[
                "token_preprocessed_embedding_source"],
            self.preprocessed_dataset[
                "translated_to_source_target"])
        self.feature_dataframe["jaccard_translation_target"] = jaccard(
            self.preprocessed_dataset[
                "token_preprocessed_embedding_target"],
            self.preprocessed_dataset[
                "translated_to_target_source"])

        # self.feature_dataframe["jaccard_named_entities_source"] = jaccard(self.preprocessed_dataset[
        #                                                                            "list_named_entities_source"],
        #                                                                self.preprocessed_dataset[
        #
        #                                                                   "named_entities_translated_to_source_target"])
        # self.feature_dataframe["jaccard_named_entities_target"] = jaccard(self.preprocessed_dataset[
        #                                                                           "list_named_entities_target"],
        #                                                               self.preprocessed_dataset[
        #
        #                                                                   "named_entities_translated_to_target_source"])

        self.feature_dataframe["jaccard_numbers_source"] = jaccard(
            self.preprocessed_dataset[
                "list_named_numbers_source"],
            self.preprocessed_dataset[
                "list_named_numbers_target"])

        self.feature_dataframe["jaccard_numbers_source"] = jaccard(
            self.preprocessed_dataset[
                "list_named_numbers_source"],
            self.preprocessed_dataset[
                "list_named_numbers_target"])

        for i in range(self.k):
            self.feature_dataframe[f"pca_embeddding_average_diff_{i}"] = embedding_difference(
                self.preprocessed_dataset[
                    "pca_sentence_embedding_average_source"],
                self.preprocessed_dataset[
                    "pca_sentence_embedding_average_target"],
                i)
            self.feature_dataframe[f"pca_embeddding_tf_idf_diff_{i}"] = embedding_difference(
                self.preprocessed_dataset[
                    "pca_sentence_embedding_tf_idf_source"],
                self.preprocessed_dataset[
                    "pca_sentence_embedding_tf_idf_target"],
                i)

        self.feature_dataframe["Translation"] = self.preprocessed_dataset["Translation"]

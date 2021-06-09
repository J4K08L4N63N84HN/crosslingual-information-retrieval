""" Class for creating features from preprocessed data.
"""

import pandas as pd

from src.features.embedding_features import cosine_similarity_vector, jaccard, euclidean_distance_vector
from src.features.sentence_features import difference_numerical, relative_difference_numerical, \
    normalized_difference_numerical
from src.utils import timer


class FeatureGeneration:
    """ Class for generating features from preprocessed parallel sentences.

    Attributes:
        preprocessed_dataset (dataframe): Preprocessed dataset.
        feature_dataframe (dataframe): Dataset containing feature for training a model.
        feature_difference_list (list): List of preprocessed columns that should be compared.
        k (int): Number of principal components.
    """

    def __init__(self, dataset_index, preprocessed_dataset):
        """ Initialize dataframe by importing preprocessed dataset..

            Args:
                dataset (str): Path of the europarl source dataset.
                embedding_list (list): List of used embeddings.
                k (int): Number of principal components.
        """
        self.dataset_index = dataset_index
        self.preprocessed_source = preprocessed_dataset.filter(regex='source$', axis=1)
        self.preprocessed_target = preprocessed_dataset.filter(regex='target$', axis=1)
        self.preprocessed_dataset = self.dataset_index.merge(self.preprocessed_source, how='left',
                                                               on="id_source").merge(self.preprocessed_target,
                                                                                     how='left',
                                                                                     on="id_target")
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
            'number_Pres', 'number_Past', 'number_']

    @timer
    def create_feature_dataframe(self):
        """ Initialize feature dataframe
        """
        self.feature_dataframe = self.dataset_index
        self.feature_dataframe["Translation"] = (
                self.feature_dataframe.id_source == self.feature_dataframe.id_target).astype(int)

    @timer
    def create_sentence_features(self):
        """ Create sentence based features.
        """

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

        self.feature_dataframe["jaccard_numbers_source"] = jaccard(
            self.preprocessed_dataset[
                "list_named_numbers_source"],
            self.preprocessed_dataset[
                "list_named_numbers_target"])

    @timer
    def create_embedding_features(self, embedding):
        """ Create embedding based features.
        """

        self.feature_dataframe[f"cosine_similarity_average_{embedding}"] = cosine_similarity_vector(
            self.preprocessed_dataset[
                f"sentence_embedding_average_{embedding}_source"],
            self.preprocessed_dataset[
                f"sentence_embedding_average_{embedding}_target"])

        self.feature_dataframe[f"cosine_similarity_tf_idf_{embedding}"] = cosine_similarity_vector(
            self.preprocessed_dataset[
                f"sentence_embedding_tf_idf_{embedding}_source"],
            self.preprocessed_dataset[
                f"sentence_embedding_tf_idf_{embedding}_target"])

        self.feature_dataframe[f"euclidean_distance_average_{embedding}"] = euclidean_distance_vector(
            self.preprocessed_dataset[
                f"sentence_embedding_average_{embedding}_source"],
            self.preprocessed_dataset[
                f"sentence_embedding_average_{embedding}_target"])

        self.feature_dataframe[f"euclidean_distance_tf_idf_{embedding}"] = euclidean_distance_vector(
            self.preprocessed_dataset[
                f"sentence_embedding_tf_idf_{embedding}_source"],
            self.preprocessed_dataset[
                f"sentence_embedding_tf_idf_{embedding}_target"])

        self.feature_dataframe[f"jaccard_translation_{embedding}"] = (jaccard(
            self.preprocessed_dataset[
                "token_preprocessed_embedding_source"],
            self.preprocessed_dataset[
                f"translated_to_source_{embedding}_target"]) + jaccard(
            self.preprocessed_dataset[
                "token_preprocessed_embedding_target"],
            self.preprocessed_dataset[
                f"translated_to_target_{embedding}_source"]))/2

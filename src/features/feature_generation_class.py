""" Class for creating a parallel sentence dataset.
"""

import pandas as pd
from src.features.sentence_based import difference_numerical


class FeatureGeneration:
    def __init__(self, preprocessed_parallel_dataset):
        self.preprocessed_dataframe = preprocessed_parallel_dataset
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
            'number_characters', 'number_ADJ', 'number_ADP',
            'number_ADV', 'number_AUX', 'number_CONJ',
            'number_CCONJ', 'number_DET', 'number_INTJ',
            'number_NOUN', 'number_NUM', 'number_PRT',
            'number_PRON', 'number_PROPN', 'number_SCONJ',
            'number_SYM', 'number_VERB', 'number_X',
            'number_Pres', 'number_Past', 'number_',
            'score_polarity', 'score_subjectivity',
            'number_stopwords']

    def feature_generation(self):
        for feature in self.feature_difference_list:
            self.feature_dataframe[f"{feature}_difference"] = difference_numerical(
                self.preprocessed_dataframe[f"{feature}_source"], self.preprocessed_dataframe[f"{feature}_target"])
            self.feature_dataframe[f"{feature}_difference_relative"] = difference_numerical(
                self.preprocessed_dataframe[f"{feature}_source"], self.preprocessed_dataframe[f"{feature}_target"])
            self.feature_dataframe[f"{feature}_difference_normalized"] = difference_numerical(
                self.preprocessed_dataframe[f"{feature}_source"], self.preprocessed_dataframe[f"{feature}_target"],
                self.preprocessed_dataframe["number_characters_source"], self.preprocessed_dataframe["number_characters_target"])
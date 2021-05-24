import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, preprocessed_class):
        self.preprocessed_dataframe = preprocessed_class.preprocessed_dataframe
        self.preprocessed_source = preprocessed_class.preprocessed_source
        self.preprocessed_target = preprocessed_class.preprocessed_target
        self.dataset = pd.DataFrame
        self.testset = pd.DataFrame()
        self.query = pd.DataFrame()
        self.documents = pd.DataFrame()
        self.feature_list = [
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
            'number_ADJ', 'number_ADP',
            'number_ADV', 'number_AUX', 'number_CONJ',
            'number_CCONJ', 'number_DET', 'number_INTJ',
            'number_NOUN', 'number_NUM', 'number_PRT',
            'number_PRON', 'number_PROPN', 'number_SCONJ',
            'number_SYM', 'number_VERB', 'number_X',
            'number_Pres', 'number_Past', 'number_',
            'score_polarity', 'score_subjectivity',
            'number_stopwords']

    def get_sample(self, n):
        """ Method to generate a sample set of 2n with n correct examples and n wrong examples.

               Args:
                   n (int): amount

               Returns:
                   dataframe: sample of 2n

               """
        # create query and document frames
        for feature in self.feature_list:
            self.query[f"{feature}_source"] = self.preprocessed_dataframe[f"{feature}_source"]
            self.documents[f"{feature}_target"] = self.preprocessed_dataframe[f"{feature}_target"]
        self.query = self.query.iloc[:5]
        self.documents = self.documents.iloc[:10]
        # generate a cross set of the queries and documents
        self.testset = self.query.reset_index().merge(self.documents.reset_index(), how='cross')
        # label with 1 if its right translation and 0 for the wrong translation
        for index_label, row_series in self.testset.iterrows():
            if self.testset.at[index_label, 'index_x'] == self.testset.at[index_label, 'index_y']:
                self.testset.at[index_label, 'Translation'] = 1
            else:
                self.testset.at[index_label, 'Translation'] = 0
        self.testset.drop(columns=['index_x', 'index_y'], inplace=True)
        self.preprocessed_dataframe.drop(self.preprocessed_dataframe.index[:3], inplace=True)
        self.preprocessed_source.drop(self.preprocessed_source.index[:3], inplace=True)
        self.preprocessed_target.drop(self.preprocessed_target.index[:3], inplace=True)
        random_sample_right = self.preprocessed_dataframe.sample(n).reset_index(drop=True)
        random_sample_wrong = pd.concat([self.preprocessed_source.sample(n).reset_index(drop=True),
                                         self.preprocessed_target.sample(n).reset_index(drop=True)],
                                        axis=1)
        random_sample_wrong["Translation"] = np.zeros((int(random_sample_wrong.shape[0]), 1),
                                                      dtype=np.int8)

        self.dataset = pd.concat([random_sample_right, random_sample_wrong]).reset_index(drop=True)

import numpy as np
import pandas as pd


class DataSet:
    """ Class for creating a modeling dataset based on the preprocessed europarl data.

    Attributes:
        preprocessed_dataframe (dataframe): Preprocessed dataframe from source and target
        preprocessed_source (dataframe): Preprocessed dataframe for source sentences
        preprocessed_target (dataframe): Preprocessed dataframe for target sentences
        dataset(dataframe): Preprocessed dataset with translated and random parallel sentences
    """
    def __init__(self, preprocessed_class):
        """ Initialize class by importing data from preprocessed class

        Args:
            preprocessed_class (class): Path of the europarl source dataset
        """
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
        """ Method to generate a training set of 2n with n correct examples and n wrong examples and a
        independent testset of 1000 queries and 10.000 documents as a crossjoin.

               Args:
                   n (int): amount

               """
        # create query and document frames
        for feature in self.feature_list:
            self.query[f"{feature}_source"] = self.preprocessed_dataframe[f"{feature}_source"]
            self.documents[f"{feature}_target"] = self.preprocessed_dataframe[f"{feature}_target"]
        self.query = self.query.iloc[:1000]
        self.documents = self.documents.iloc[:10000]
        # generate a cross set of the queries and documents
        self.testset = self.query.reset_index().merge(self.documents.reset_index(), how='cross')
        # label with 1 if its right translation and 0 for the wrong translation
        for index_label, row_series in self.testset.iterrows():
            if self.testset.at[index_label, 'index_x'] == self.testset.at[index_label, 'index_y']:
                self.testset.at[index_label, 'Translation'] = 1
            else:
                self.testset.at[index_label, 'Translation'] = 0
        self.testset.drop(columns=['index_x', 'index_y'], inplace=True)
        # drop first 10.000 documents from train set to make sure that its independent
        self.preprocessed_dataframe.drop(self.preprocessed_dataframe.index[:10000], inplace=True)
        self.preprocessed_source.drop(self.preprocessed_source.index[:10000], inplace=True)
        self.preprocessed_target.drop(self.preprocessed_target.index[:10000], inplace=True)
          Args:
              n (int): Number of correct and incorrect sentence pairs
          """
        random_sample_right = self.preprocessed_dataframe.sample(n).reset_index(drop=True)
        random_sample_wrong = pd.concat([self.preprocessed_source.sample(n).reset_index(drop=True),
                                         self.preprocessed_target.sample(n).reset_index(drop=True)],
                                        axis=1)
        random_sample_wrong["Translation"] = np.zeros((int(random_sample_wrong.shape[0]), 1),
                                                      dtype=np.int8)

        self.dataset = pd.concat([random_sample_right, random_sample_wrong]).reset_index(drop=True)

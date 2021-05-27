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
        self.preprocessed_dataframe = preprocessed_class.preprocessed
        self.preprocessed_source = preprocessed_class.preprocessed.filter(regex='source$', axis=1)
        self.preprocessed_target = preprocessed_class.preprocessed.filter(regex='target$', axis=1)
        self.dataset = pd.DataFrame
        self.testset = pd.DataFrame()
        self.query = pd.DataFrame()
        self.documents = pd.DataFrame()

    def get_sample(self, n_training, n_test_queries, n_test_documents):
        """ Method to generate a training set of 2n with n correct examples and n wrong examples and a
        independent testset of 1000 queries and 10.000 documents as a crossjoin.

               Args:

              n (int): Number of correct and incorrect sentence pairs

               """
        # create query and document frames
        self.query = self.preprocessed_dataframe.filter(regex='source$', axis=1).iloc[:n_test_queries]
        self.documents = self.preprocessed_dataframe.filter(regex='target$', axis=1).iloc[:n_test_documents]
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
        self.preprocessed_dataframe.drop(self.preprocessed_dataframe.index[:n_test_documents], inplace=True)
        self.preprocessed_source.drop(self.preprocessed_source.index[:n_test_queries], inplace=True)
        self.preprocessed_target.drop(self.preprocessed_target.index[:n_test_documents], inplace=True)

        random_sample_right = self.preprocessed_dataframe.sample(n_training).reset_index(drop=True)
        random_sample_wrong = pd.concat([self.preprocessed_source.sample(n_training).reset_index(drop=True),
                                         self.preprocessed_target.sample(n_training).reset_index(drop=True)],
                                        axis=1)
        random_sample_wrong["Translation"] = np.zeros((int(random_sample_wrong.shape[0]), 1),
                                                      dtype=np.int8)

        self.dataset = pd.concat([random_sample_right, random_sample_wrong]).reset_index(drop=True)

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
        self.preprocessed_source = preprocessed_class.preprocessed.filter(regex='source$|id$', axis=1).rename(
            columns={"id": "id_source"})
        self.preprocessed_target = preprocessed_class.preprocessed.filter(regex='target$|id$', axis=1).rename(
            columns={"id": "id_target"})
        self.dataset = pd.DataFrame
        self.testset = pd.DataFrame()
        self.query = pd.DataFrame()
        self.documents = pd.DataFrame()

    def get_sample(self, n_training, n_test_queries, n_test_documents, k):
        """ Method to generate a training set of 2n with n correct examples and n wrong examples and a
        independent testset of 1000 queries and 10.000 documents as a crossjoin.

               Args:

              n (int): Number of correct and incorrect sentence pairs

               """
        self.query = self.preprocessed_dataframe.filter(regex='source$|id$', axis=1).iloc[:n_test_queries].rename(
            columns={"id": "id_source"})
        self.documents = self.preprocessed_dataframe.filter(regex='target$|id$', axis=1).iloc[
                            :n_test_documents].rename(
            columns={"id": "id_target"})

        self.testset = self.query.reset_index().merge(self.documents.reset_index(), how='cross')

        self.testset["Translation"] = (self.testset["id_source"] == self.testset["id_target"]).astype("int")

        random_sample_right = self.preprocessed_dataframe.iloc[n_test_queries:].sample(n_training).reset_index(
            drop=True)
        random_sample_right.rename(columns={"id": "id_source"}, inplace = True)
        random_sample_right["id_target"] = random_sample_right["id_source"]

        random_sample_wrong = pd.concat(
            [self.preprocessed_source.iloc[n_test_queries:].sample(k * n_training).reset_index(drop=True),
             self.preprocessed_target.iloc[n_test_documents:].sample(
                 k * n_training, replace=True).reset_index(drop=True)],
            axis=1)
        random_sample_wrong["Translation"] = (
                    random_sample_wrong["id_source"] == random_sample_wrong["id_target"]).astype("int")

        self.dataset = pd.concat([random_sample_right, random_sample_wrong]).reset_index(drop=True)

""" Class for creating datasets for training a supervised translation classifier and for crosslingual information
retrieval.
"""

import pandas as pd
from tqdm import tqdm

from src.features.embedding_features import cosine_similarity_vector
from src.utils.timer import timer


class DataSet:
    """ Class for creating datasets for training a supervised translation classifier and for crosslingual information
        retrieval.

    Attributes:
        preprocessed_dataframe (dataframe): Preprocessed parallel translations dataframe.
        model_subset (dataframe): Subset of preprocessed_dataframe for training a supervised translation classifier.
        retrieval_subset (dataframe): Subset of preprocessed_datafram for testing crosslingual retrieval models.
        model_dataset (dataframe):Generated dataset for training a supervised translation classifier.
        retrieval_dataset (dataframe): Dataset for testing crosslingual retrieval models.
    """

    @timer
    def __init__(self, preprocessed_data):
        """ Initialize class by importing preprocessed data.

            Args:
                preprocessed_data (dataframe): Preprocessed dataframe of parallel translations.
        """
        self.preprocessed_dataframe = preprocessed_data
        self.model_subset = pd.DataFrame()
        self.retrieval_subset = pd.DataFrame()
        self.model_dataset_index = pd.DataFrame()
        self.model_dataset = pd.DataFrame()
        self.retrieval_dataset = pd.DataFrame()
        self.retrieval_dataset_index = pd.DataFrame()

    @timer
    def split_model_retrieval(self, n_model=20000, n_retrieval=5000):
        """ Split data into model dataset and retrieval dataset.

            Args:
                n_model (int): Number of preprocessed datapoints used for supervised modelling.
                n_retrieval (int): Number of preprocessed datapoints used for the retrieval task.
        """
        try:
            self.model_subset = self.preprocessed_dataframe.iloc[0:n_model]
            self.retrieval_subset = self.preprocessed_dataframe.iloc[n_model:(n_model + n_retrieval)]

        except IndexError:
            print("n_model + n_retrieval must be smaller than the dataset size.")

    def create_model_index(self, n_model=5000, k=5, sample_size_k=100,
                           embedding_source="sentence_embedding_tf_idf_proc_5k_source",
                           embedding_target="sentence_embedding_tf_idf_proc_5k_target"):
        """ Generate dataset for modelling a supervised classifier.

            Args:
                n_model (int): Number of preprocessed datapoints used for supervised modelling.
                k (int): Number of false translated sentences pair for training a supervised classifier.
                sample_size_k (int): Number of samples from target per source sentence for searching nearest sentences.
                embedding_source (str): Name of source embeddings
                embedding_target (str): Name of source embeddings

        """

        preprocessed_source = self.model_subset[["id_source", embedding_source]]
        preprocessed_target = self.model_subset[["id_target", embedding_target]]

        random_sample_right = self.model_subset[["id_source", "id_target"]]

        multiplied_source = pd.concat([preprocessed_source] * sample_size_k, ignore_index=True).reset_index(
            drop=True)
        sample_target = preprocessed_target.sample(n_model * sample_size_k, replace=True, random_state=42).reset_index(
            drop=True)

        random_sample_wrong = pd.concat([multiplied_source, sample_target], axis=1)

        # Select only the 2*k closest sentence embeddings for training to increase the complexity of the task for
        # the supervised classifier.
        random_sample_wrong["cosine_similarity"] = cosine_similarity_vector(
            random_sample_wrong["sentence_embedding_tf_idf_proc_5k_source"],
            random_sample_wrong["sentence_embedding_tf_idf_proc_5k_target"])

        random_sample_k_index = random_sample_wrong.groupby("id_source")['cosine_similarity'].nlargest(k)

        rows = []
        for i in tqdm(range(n_model)):
            for key in random_sample_k_index[i].keys():
                rows.append(key)
        random_sample_k = random_sample_wrong.iloc[rows].reset_index(drop=True)[["id_source", "id_target"]]

        self.model_dataset_index = pd.concat([random_sample_right, random_sample_k], axis=0).reset_index(drop=True)

    @timer
    def create_retrieval_index(self, n_queries):
        """ Generate dataset for modelling a supervised classifier.

            Args:
                n_queries (int): Number of source sentences used as queries.
        """
        # Select the first n_queries since data was already sampled in the start.
        query = pd.DataFrame({"id_source": self.retrieval_subset.iloc[:n_queries]["id_source"]})

        documents = pd.DataFrame({"id_target": self.retrieval_subset["id_target"]})

        self.retrieval_dataset_index = query.merge(documents, how='cross')

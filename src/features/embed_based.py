""" Functions to work with the word embeddings
"""

import numpy as np


def sentence_embedding(token_vector, embedding_matrix_path, embedding_dictionary_path):
        """ Function to generate ***

                           Args:
                               token_vector (numpy.array):
                               embedding_matrix_path (numpy.matrix):
                               embedding_dictionary_path

                           Returns:
                               xxx

                           """

        embedding_sentence = np.delete(embedding_sentence, deletion_list, axis=0)
        embedding_sentence_mean = np.mean(embedding_sentence, axis=0)
        return embedding_sentence_mean
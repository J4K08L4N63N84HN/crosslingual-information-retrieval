""" Functions to create features based on crosslingual word embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm
from wmd import WMD

from src.data.preprocess_data import timer

tqdm.pandas()


@timer
def cosine_similarity_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    """ Calculate cosine similarity of sentece embeddings.
        Args:
            sentence_embedding_source_vector (array): Array of sentence embeddings in source language.
            sentence_embedding_target_vector (array):Array of sentence embeddings in target language.

        Returns:
            array: Array containing cosine similarity measures.
    """
    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def cosine_similarity_pairwise(sentence_embedding_source, sentence_embedding_target):
        """ Calculate cosine similarity between two arrays.
        """
        try:
            sentence_embedding_source_array = np.array(list(sentence_embedding_source[0].values())).reshape(1, -1)
            sentence_embedding_target_array = np.array(list(sentence_embedding_target[0].values())).reshape(1, -1)
            try:
                return cosine_similarity(X=sentence_embedding_source_array, Y=sentence_embedding_target_array,
                                         dense_output=True)[0][0]
            except ValueError:
                return 0
        except TypeError:
            sentence_embedding_source_array = np.array(sentence_embedding_source).reshape(1, -1)
            sentence_embedding_target_array = np.array(sentence_embedding_target).reshape(1, -1)
            try:
                return cosine_similarity(X=sentence_embedding_source_array, Y=sentence_embedding_target_array,
                                         dense_output=True)[0][0]
            except ValueError:
                return 0

    cosine_similarity_score = df.progress_apply(lambda x: cosine_similarity_pairwise(x.sentence_embedding_source,
                                                                                     x.sentence_embedding_target),
                                                axis=1)

    return cosine_similarity_score


@timer
def euclidean_distance_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    """ Calculate euclidean distance of sentece embeddings.

        Args:
            sentence_embedding_source_vector (array): Array of sentence embeddings in source language.
            sentence_embedding_target_vector (array):Array of sentence embeddings in target language.

        Returns:
            array: Array containing euclidean distance measures.
    """

    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def euclidean_distance_pairwise(sentence_embedding_source, sentence_embedding_target):
        """ Calculate euclidean distance between two arrays.
        """
        try:
            sentence_embedding_source_array = np.array(list(sentence_embedding_source[0].values())).reshape(1, -1)
            sentence_embedding_target_array = np.array(list(sentence_embedding_target[0].values())).reshape(1, -1)
            try:
                return euclidean_distances(X=sentence_embedding_source_array, Y=sentence_embedding_target_array)[0][0]
            except ValueError:
                return 0
        except TypeError:
            sentence_embedding_source_array = np.array(sentence_embedding_source).reshape(1, -1)
            sentence_embedding_target_array = np.array(sentence_embedding_target).reshape(1, -1)
            try:
                return euclidean_distances(X=sentence_embedding_source_array, Y=sentence_embedding_target_array)[0][0]
            except ValueError:
                return 0

    euclidean_distance_score = df.progress_apply(lambda x: euclidean_distance_pairwise(x.sentence_embedding_source,
                                                                                       x.sentence_embedding_target),
                                                 axis=1)

    return euclidean_distance_score


@timer
def word_mover_distance_vector(word_embedding_source_vector, word_embedding_target_vector):
    """ Calculate word mover distance between word embeddings of two sentences.
        Args:
            word_embedding_source_vector (array): Array of word embeddings of sentences in source language.
            word_embedding_target_vector (array): Array of word embeddings of sentences in target language.

        Returns:
            array: Array containing word mover distance measures.
    """
    df = pd.DataFrame()
    df['word_embedding_source'] = word_embedding_source_vector
    df['word_embedding_target'] = word_embedding_target_vector

    def word_mover_distance(word_embedding_dict_source, word_embedding_dict_target):
        """ Calculate euclidean distance between two dictionaries of arrays.
        """
        try:
            source = np.array(word_embedding_dict_source, dtype=np.float32)
            target = np.array(word_embedding_dict_target, np.float32)
            embeddings = np.concatenate((source, target))

            source_len = source.shape[0]
            target_len = target.shape[0]

            source_words = np.array([i for i in range(source_len)], dtype=np.int32)
            target_words = np.array([source_len + i for i in range(target_len)], dtype=np.int32)

            source_weights = np.array([1 for i in range(source_len)], dtype=np.int32)
            target_weights = np.array([1 for i in range(target_len)], dtype=np.int32)

            nbow = {"source": ("source", source_words, source_weights),
                    "target": ("target", target_words, target_weights)}
            calc = WMD(embeddings, nbow, vocabulary_min=2)

            return calc.nearest_neighbors("source", 1)[0][1]

        except (ValueError, IndexError):
            return 0

    word_mover_distance_score = df.progress_apply(lambda x: word_mover_distance(x.word_embedding_source,
                                                                                x.word_embedding_target),
                                                  axis=1)

    return word_mover_distance_score


@timer
def jaccard(vector_source, vector_target):
    """ Calculate jaccard similarity between two list of tokens.
        Args:
            vector_source (array): Array of preprocessed token in source language.
            vector_target (array): Array of preprocessed token in target language.

        Returns:
            array: Array containing jaccard similarity.
    """
    df = pd.DataFrame()
    df['original'] = vector_source
    df['translated'] = vector_target

    def jaccard_similarity_score(original, translation):
        """ Calculate jaccard similarity between two lists.
        """
        intersect = set(original).intersection(set(translation))
        union = set(original).union(set(translation))
        try:
            return len(intersect) / len(union)
        except ZeroDivisionError:
            return 0

    jaccard_vec = df.progress_apply(lambda x: jaccard_similarity_score(x.original,
                                                                       x.translated),
                                    axis=1)
    return jaccard_vec


@timer
def embedding_difference(pca_sentence_embedding_source, pca_sentence_embedding_target, i):
    """ Calculate embedding difference for pca sentence embeddings.

        Args:
            pca_sentence_embedding_source (array): Array of pca sentence embeddings in source language.
            pca_sentence_embedding_target (array):Array of pca sentence embeddings in target language.
            i (int): Position in the arrays.

        Returns:
            array: Array embedding differences.
    """
    df = pd.DataFrame()
    df['source'] = pca_sentence_embedding_source
    df['target'] = pca_sentence_embedding_target

    def difference(embedding_source, embedding_target, i):
        """ Calculate difference between array at position i.
        """
        try:
            embedding_source_row = embedding_source[0][i]
        except (KeyError, IndexError):
            embedding_source_row = 0

        try:
            embedding_target_row = embedding_target[0][i]
        except (KeyError, IndexError):
            embedding_target_row = 0

        return embedding_source_row - embedding_target_row

    diff_vec = df.progress_apply(lambda x: difference(x.source, x.target, i),
                                 axis=1)
    return diff_vec

""" Functions to work with the word embeddings
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from wmd import WMD


def cosine_similarity_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def cosine_similarity_pairwise(sentence_embedding_source, sentence_embedding_target):
        try:
            return cosine_similarity(X=sentence_embedding_source, Y=sentence_embedding_target, dense_output=True)[0][0]
        except ValueError:
            return 0

    cosine_similarity_score = df.apply(lambda x: cosine_similarity_pairwise(x.sentence_embedding_source,
                                                                            x.sentence_embedding_target),
                                       axis=1)

    return cosine_similarity_score


def euclidean_distance_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def euclidean_distance_pairwise(sentence_embedding_source, sentence_embedding_target):
        try:
            return euclidean_distances(X=sentence_embedding_source, Y=sentence_embedding_target)[0][0]
        except ValueError:
            return 0

    euclidean_distance_score = df.apply(lambda x: euclidean_distance_pairwise(x.sentence_embedding_source,
                                                                              x.sentence_embedding_target),
                                        axis=1)

    return euclidean_distance_score


def word_mover_distance_vector(word_embedding_source_vector, word_embedding_target_vector):
    df = pd.DataFrame()
    df['word_embedding_source'] = word_embedding_source_vector
    df['word_embedding_target'] = word_embedding_target_vector

    def word_mover_distance(word_embedding_dataframe_source, word_embedding_dataframe_target):
        try:
            source = np.array(word_embedding_dataframe_source.values.transpose(), dtype=np.float32)
            target = np.array(word_embedding_dataframe_target.values.transpose(), dtype=np.float32)
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
        except (ValueError,IndexError):
            return 0

    word_mover_distance_score = df.apply(lambda x: word_mover_distance(x.word_embedding_source,
                                                                       x.word_embedding_target),
                                         axis=1)

    return word_mover_distance_score


def jaccard(vector_source, vector_target):
    df = pd.DataFrame()
    df['original'] = vector_source
    df['translated'] = vector_target

    def jaccard_similarity_score(original, translation):
        intersect = set(original).intersection(set(translation))
        union = set(original).union(set(translation))
        try:
            return len(intersect) / len(union)
        except ZeroDivisionError:
            return 0

    jaccard_vec = df.apply(lambda x: jaccard_similarity_score(x.original,
                                                              x.translated),
                           axis=1)
    return jaccard_vec


def embedding_difference(vector_embedding_source, vector_embedding_target, i):
    df = pd.DataFrame()
    df['source'] = vector_embedding_source
    df['target'] = vector_embedding_target

    def difference(embedding_source, embedding_target, i):
        try:
            embedding_source_row = embedding_source[0][i]
        except (KeyError, IndexError):
            embedding_source_row = 0

        try:
            embedding_target_row = embedding_target[0][i]
        except (KeyError, IndexError):
            embedding_target_row = 0

        return embedding_source_row - embedding_target_row

    diff_vec = df.apply(lambda x: difference(x.source, x.target, i),
                        axis=1)
    return diff_vec

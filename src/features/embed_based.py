""" Functions to work with the word embeddings
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def cosine_similarity_pairwaise(sentence_embedding_source, sentence_embedding_target):
        try:
            return cosine_similarity(X=sentence_embedding_source, Y=sentence_embedding_target, dense_output=True)[0][0]
        except ValueError:
            return 0

    consine_similarity_score = df.apply(lambda x: cosine_similarity_pairwaise(x.sentence_embedding_source,
                                                                              x.sentence_embedding_target),
                                        axis=1)

    return consine_similarity_score


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

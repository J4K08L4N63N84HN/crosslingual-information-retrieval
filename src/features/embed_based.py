""" Functions to work with the word embeddings
"""

from sklearn.metrics import jaccard_score

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_vector(sentence_embedding_source_vector, sentence_embedding_target_vector):
    df = pd.DataFrame()
    df['sentence_embedding_source'] = sentence_embedding_source_vector
    df['sentence_embedding_target'] = sentence_embedding_target_vector

    def cosine_similarity_pairwaise(sentence_embedding_source, sentence_embedding_target):
        return cosine_similarity(X=sentence_embedding_source, Y=sentence_embedding_target, dense_output=True)[0][0]

    consine_similarity_score = df.apply(lambda x: cosine_similarity_pairwaise(x.sentence_embedding_source,
                                                                              x.sentence_embedding_target),
                                        axis=1)

    return consine_similarity_score


def jaccard(original_vector, translated_vector):
    df = pd.DataFrame()
    df['original'] = original_vector
    df['translated'] = translated_vector

    def jaccard_similarity_score(original, translation):
        intersect = set(original).intersection(set(translation))
        union = set(original).union(set(translation))
        return len(intersect) / len(union)

    jaccard_vec = df.apply(lambda x: jaccard_similarity_score(x.original,
                                                              x.translated),
                           axis=1)
    return jaccard_vec
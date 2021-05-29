""" Functions to train a unsupervised retrieval model

TODO: Build unsupervised model with cosine similarity
TODO: Build unsupervised model with euclidean distance
TODO: Build unsupervised model with word mover distance
"""

from src.data.preprocess_data import timer

@timer
def unsupervised_retrieval(dataframe, id_source, k):
    df_filter = dataframe[dataframe.id_source.eq(id_source)]
    df_sorted = df_filter.sort_values("cosine_similarity_average", axis=1)
    df_sorted_k = df_sorted.iloc[:k]
    return df_sorted_k.id_target

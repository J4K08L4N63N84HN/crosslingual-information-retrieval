"""
.. automodule:: src.features.embed_based
    :members:
.. automodule:: src.data.feature_generation_class
    :members:
.. automodule:: src.features.sentence_based
    :members:
"""

from .embed_based import cosine_similarity_vector, euclidean_distance_vector, word_mover_distance_vector, jaccard, \
    embedding_difference
from .feature_generation_class import FeatureGeneration
from .sentence_based import difference_numerical, relative_difference_numerical, normalized_difference_numerical

"""
.. automodule:: src.embeddings.evaluation_bli
    :members:
.. automodule:: src.embeddings.load_monolingual
    :members:
.. automodule:: src.embeddings.modify_dictionary
    :members:
.. automodule:: src.embeddings.pipeline_clwe_induction
    :members:
.. automodule:: src.embeddings.supervised_cle
    :members:
.. automodule:: src.embeddings.text_encoders
    :members:
.. automodule:: src.embeddings.unsupervised_cle
    :members:
.. automodule:: src.embeddings.utils
    :members:
"""
from .evaluation_bli import Evaluator
from .load_monolingual import load_embedding, load_translation_dict, save_clew
from .modify_dictionary import cut_dictionary_to_vocabulary
from .pipeline_clwe_induction import clew_induction
from .supervised_cle import Projection_based_clwe
from .text_encoders import TextEncoders
from .unsupervised_cle import VecMap
from .utils import supports_cupy, get_cupy, get_array_module, asnumpy, find_nearest_neighbor, normalize_matrix, \
    check_if_neighbors_match, mean_center, vecmap_normalize, topk_mean, dropout, big_matrix_multiplication

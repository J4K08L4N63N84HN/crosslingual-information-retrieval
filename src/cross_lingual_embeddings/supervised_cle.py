from src.cross_lingual_embeddings.load_monolingual import load_translation_dict, load_embedding
from src.cross_lingual_embeddings.utils import normalize_matrix, check_if_neighbors_match, find_nearest_neighbor
import numpy as np


class Projection_based_clwe:
    # Embeddings
    source_embedding_matrix = []
    target_embedding_matrix = []

    # Word to index map
    src_word2ind = {}
    trg_word2ind = {}

    # Index to word map
    src_ind2word = {}
    trg_ind2word = {}

    # Normalized Embeddings (need for calculating similarities)
    norm_src_embedding_matrix = []
    norm_trg_embedding_matrix = []

    # Train Translation Dictionary
    train_translation_source = []
    train_translation_target = []
    test_translation_source = []
    test_translation_target = []

    # Created Subspace
    X_source_embedding = []
    X_target_embedding = []

    # Mappings W
    mapping_source_target = []
    mapping_target_source = []

    # Projected Embeddings
    proj_embedding_source_target = []
    proj_embedding_target_source = []

    def __init__(self,
                 path_source_language, path_target_language,
                 train_translation_dict_path
                 ):

        # Built Embeddings
        self.source_embedding_word, self.source_embedding_matrix = load_embedding(path_source_language)
        self.target_embedding_word, self.target_embedding_matrix = load_embedding(path_target_language)

        # Built train/test dictionary
        self.train_translation_source, self.train_translation_target = load_translation_dict(
            train_translation_dict_path)

        # Built Word to index map
        self.src_word2ind = {word: i for i, word in enumerate(self.source_embedding_word)}
        self.trg_word2ind = {word: i for i, word in enumerate(self.target_embedding_word)}

        # Built Index to Word map
        self.src_ind2word = {i: word for i, word in enumerate(self.source_embedding_word)}
        self.trg_ind2word = {i: word for i, word in enumerate(self.target_embedding_word)}

        # Normalize Embeddings
        self.norm_src_embedding_matrix = normalize_matrix(self.source_embedding_matrix)
        self.norm_trg_embedding_matrix = normalize_matrix(self.target_embedding_matrix)

    def proc_bootstrapping(self, n_iterations=10):
        print("Length of Original dictionary: {}".format(len(self.train_translation_source)))
        for i in range(n_iterations):
            self.get_subspace()
            self.solve_proscrutes_problem(source_to_target=True)
            self.solve_proscrutes_problem(source_to_target=False)
            # No need to augment dictionary, if its last iteration
            if i == n_iterations - 1:
                break

            # Project Source embedding to Target Embedding
            self.project_embedding_space(source_to_target=True)
            # Project Target embedding to Source Embedding
            self.project_embedding_space(source_to_target=False)
            # Start Augmenting Dictionary
            self.augment_dictionary()

            print("Length of new dictionary: {}".format(len(self.train_translation_source)))

    def proc(self, source_to_target):

        self.get_subspace()
        self.solve_proscrutes_problem(source_to_target)
        self.project_embedding_space(source_to_target)

    def augment_dictionary(self):

        # Find NN from projected source to (original) target embedding
        neighbors_projected_src_trg = find_nearest_neighbor(
            normalize_matrix(self.proj_embedding_source_target),
            self.norm_trg_embedding_matrix)

        # Find NN from projected target embedding to (original) source embedding
        neighbors_projected_trg_src = find_nearest_neighbor(
            normalize_matrix(self.proj_embedding_target_source),
            self.norm_src_embedding_matrix)

        # Find Matches
        matching = check_if_neighbors_match(neighbors_projected_src_trg,
                                            neighbors_projected_trg_src)

        # Update orignal Dictionary
        self.train_translation_source += [self.src_ind2word[source_index] for source_index in list(matching.keys())]
        self.train_translation_target += [self.trg_ind2word[target_index] for target_index in list(matching.values())]

    def get_subspace(self):

        index_source_embedding = []
        index_target_embedding = []

        for index_translation in range(len(self.train_translation_source)):
            source_word = self.train_translation_source[index_translation]
            target_word = self.train_translation_target[index_translation]
            if source_word not in self.src_word2ind.keys():
                continue
            if target_word not in self.trg_word2ind.keys():
                continue

            index_source_embedding.append(self.src_word2ind[source_word])
            index_target_embedding.append(self.trg_word2ind[target_word])

        self.X_source_embedding = self.source_embedding_matrix[index_source_embedding]
        self.X_target_embedding = self.target_embedding_matrix[index_target_embedding]

    def solve_proscrutes_problem(self, source_to_target):

        if source_to_target:
            U, s, V_t = np.linalg.svd(np.matmul(self.X_source_embedding.transpose(), self.X_target_embedding))
            self.mapping_source_target = np.matmul(U, V_t)
        else:
            U, s, V_t = np.linalg.svd(np.matmul(self.X_target_embedding.transpose(), self.X_source_embedding))
            self.mapping_target_source = np.matmul(U, V_t)

    def project_embedding_space(self, source_to_target):

        if source_to_target:
            self.proj_embedding_source_target = np.matmul(self.source_embedding_matrix, self.mapping_source_target)
        else:
            self.proj_embedding_target_source = np.matmul(self.target_embedding_matrix, self.mapping_target_source)

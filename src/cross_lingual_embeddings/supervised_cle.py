import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_monolingual import load_translation_dict, load_embedding
from utils import normalize_matrix, check_if_neighbors_match, \
find_nearest_neighbor, big_matrix_multiplication
import numpy as np


class Projection_based_clwe:
    """Induce CLWE with Proc-B and Proc.

   Attributes:
       source_embedding_matrix (array): Original Monolingual Source Embedding Matrix.
       target_embedding_matrix (array): Original Monolingual Source Embedding Matrix.
       src_word2ind (dict): Dictionary of source word to index.
       trg_word2ind (dict): Dictionary of target word to index.
       src_ind2word (dict): Dictionary of index to word source.
       trg_ind2word (dict): Dictionary of index to word target.
       norm_src_embedding_matrix (array): Original Normalized Monolingual Source Embedding Matrix.
       norm_trg_embedding_matrix (array): Original Normalized Monolingual Source Embedding Matrix.
       train_translation_source (dict): List of source translation words.
       train_translation_target (dict): List of target translation words.
       X_source_embedding (array): Subspace of source embedding, depending on translation dictionary.
       X_target_embedding (array): Subspace of taarget embedding, depending on translation dictionary.
       mapping_source_target (array): Source to Target Matrix Projection.
       mapping_target_source (array): Target to Source Matrix Projection.
       proj_embedding_source_target (array): Projected Source Embedding to Target Space (CLWE).
       proj_embedding_target_source (array): Projected Target Embedding to Source Space (CLWE).

   """
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
                 train_translation_dict_path, number_tokens=5000):
        """Proc and Proc-b Method Class.

        Args:
            path_source_language (path): Path to source Language.
            path_target_language (path): Path to target Language.
            train_translation_dict_path (path): Path to train translation dictionary.
            number_tokens (int): Number of tokens per language.
        """
        # Built Embeddings
        self.source_embedding_word, self.source_embedding_matrix = load_embedding(path_source_language, number_tokens)
        self.target_embedding_word, self.target_embedding_matrix = load_embedding(path_target_language, number_tokens)

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

    def proc_bootstrapping(self, growth_rate=1.5, limit=10000):
        """Function to use Proc-B method - Induces CLWE.

        Args:
            growth_rate (int): Growth rate of augmented dictionary.
            limit (int): Augmented Dictionary Limit

        Returns:

        """
        print("Length of Original dictionary: {}".format(len(self.train_translation_source)))
        size = 0
        current_iteration = 0
        while True:
            current_iteration += 1
            self.get_subspace()
            self.solve_proscrutes_problem(source_to_target=True)
            self.solve_proscrutes_problem(source_to_target=False)

            size1 = self.X_source_embedding.shape[0]
            # No need to augment dictionary, if its last iteration
            if size1 < 1.01 * size or size1 >= limit:
                break
            else:
                size = size1

            # Project Source embedding to Target Embedding
            self.project_embedding_space(source_to_target=True)
            # Project Target embedding to Source Embedding
            self.project_embedding_space(source_to_target=False)
            # Start Augmenting Dictionary
            self.augment_dictionary(growth_rate, limit)

            print("Length of new dictionary: {}".format(len(self.train_translation_source)))

    def proc(self, source_to_target):
        """Function to use Proc method - Induces CLWE.

        Args:
            source_to_target (boolean): Define the direction of projection.

        Returns:

        """
        self.get_subspace()
        self.solve_proscrutes_problem(source_to_target)
        self.project_embedding_space(source_to_target)

    def augment_dictionary(self, growth_rate, limit):
        """Augment the Dictionary based on Proc-B method.

        Args:
            growth_rate (int): Growth rate of augmented dictionary.
            limit (int): Augmented Dictionary Limit

        Returns:

        """
        # Find NN from projected source to (original) target embedding
        neighbors_projected_src_trg = find_nearest_neighbor(
            normalize_matrix(self.proj_embedding_source_target),
            self.norm_trg_embedding_matrix, use_batch=True)
        # Find NN from projected target embedding to (original) source embedding
        neighbors_projected_trg_src = find_nearest_neighbor(
            normalize_matrix(self.proj_embedding_target_source),
            self.norm_src_embedding_matrix, use_batch=True)
        # Find Matches
        matching = check_if_neighbors_match(neighbors_projected_src_trg,
                                            neighbors_projected_trg_src)
        # Make Sure that it does not grow fast
        rank_pairs = [[key, value] for key, value in matching.items()]
        cnt = min(int(growth_rate * len(self.train_translation_source)), limit)
        if cnt < len(rank_pairs):
            rank_pairs = rank_pairs[:cnt]
        # Update orignal Dictionary
        self.train_translation_source = [self.src_ind2word[source_index] for source_index in [pair[0] for pair in rank_pairs]]
        self.train_translation_target = [self.trg_ind2word[target_index] for target_index in [pair[1] for pair in rank_pairs]]

    def get_subspace(self):
        """Get the Source and target Subspace based on the provided translation dictionary.

        Returns:

        """
        print("Length of Original dictionary: {}".format(len(self.train_translation_source)))
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

        print("Length of dictionary after pruning: {}".format(len(index_target_embedding)))
        self.X_source_embedding = self.source_embedding_matrix[index_source_embedding]
        self.X_target_embedding = self.target_embedding_matrix[index_target_embedding]

    def solve_proscrutes_problem(self, source_to_target):
        """Solve the Proscrutes Problem.

        Args:
            source_to_target (boolean): Define the direction of projection.

        Returns:

        """
        if source_to_target:
            U, s, V_t = np.linalg.svd(np.matmul(self.X_source_embedding.transpose(), self.X_target_embedding))
            self.mapping_source_target = np.matmul(U, V_t)
        else:
            U, s, V_t = np.linalg.svd(np.matmul(self.X_target_embedding.transpose(), self.X_source_embedding))
            self.mapping_target_source = np.matmul(U, V_t)

    def project_embedding_space(self, source_to_target):
        
        if source_to_target:
            self.proj_embedding_source_target = big_matrix_multiplication(self.source_embedding_matrix, self.mapping_source_target)
        else:
            self.proj_embedding_target_source = big_matrix_multiplication(self.target_embedding_matrix, self.mapping_target_source)

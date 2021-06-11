import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_monolingual import load_embedding
from utils import vecmap_normalize, supports_cupy, get_cupy, topk_mean, dropout
import numpy as np
import time

try:
    import cupy
except:
    print("No cupy installed")
    cupy = None


class VecMap:
    """Induce CLWE with VecMap.

   Attributes:
       target_embedding_matrix (array): Original Monolingual Source Embedding Matrix.
       src_indices (list): Build the seed dictionary.
       trg_indices (list): Build the seed dictionary.
       src_word2ind (dict): Dictionary of source word to index.
       trg_word2ind (dict): Dictionary of target word to index.
       src_ind2word (dict): Dictionary of index to word source.
       trg_ind2word (dict): Dictionary of index to word target.
       norm_trg_embedding_matrix (array): Original Normalized Monolingual Source Embedding Matrix.
       proj_embedding_source_target (array): Projected Source Embedding to Target Space (CLWE).

   """
    # Use CSLS for dictionary induction
    csls_neighborhood = 10
    # Restrict the vocabulary to the top k entries for unsupervised initialization
    unsupervised_vocab = 4000
    # Restrict the vocabulary to the top k entries
    vocabulary_cutoff = 20000
    # Re-weight the language embeddings
    src_reweight = 0.5
    trg_reweight = 0.5
    # Apply dimensionality reduction
    dim_reduction = 0

    threshold = 0.000001
    batch_size = 1000

    # Build the seed dictionary
    src_indices = []
    trg_indices = []

    # Target Embedding
    target_embedding_matrix = []

    # Source Embeddings
    proj_embedding_source_target = []

    # Word to index map
    src_word2ind = {}
    trg_word2ind = {}

    # Index to word map
    src_ind2word = {}
    trg_ind2word = {}

    # Normalized Embeddings (need for calculating similarities)
    norm_trg_embedding_matrix = []

    def __init__(self,
                 path_source_language,
                 path_target_language, number_tokens=5000):
        """Initialize VecMap Method Class.

            Args:
                path_source_language: Path to source Language.
                path_target_language: Path to target Language.
                number_tokens: Number of tokens per language.
        """
        # Built Embeddings
        self.source_embedding_word, self.source_embedding_matrix = load_embedding(path_source_language, number_tokens)
        self.target_embedding_word, self.target_embedding_matrix = load_embedding(path_target_language, number_tokens)

        # Built Index to Word map
        self.src_ind2word = {i: word for i, word in enumerate(self.source_embedding_word)}
        self.trg_ind2word = {i: word for i, word in enumerate(self.target_embedding_word)}

        # Built Word to index map
        self.src_word2ind = {word: i for i, word in enumerate(self.source_embedding_word)}
        self.trg_word2ind = {word: i for i, word in enumerate(self.target_embedding_word)}

        # Normalize Embeddings
        self.norm_src_embedding_matrix = vecmap_normalize(self.source_embedding_matrix)
        self.norm_trg_embedding_matrix = vecmap_normalize(self.target_embedding_matrix)

    def build_seed_dictionary(self, use_gpu=False):
        """Built the Seed Dictionary.

        Args:
            use_gpu (boolean): Use GPU (recommended) or not

        Returns:

        """
        if use_gpu:
            if not supports_cupy():
                print('ERROR: Install CuPy for CUDA support')
                sys.exit(-1)
            xp = get_cupy()
        else:
            xp = np

        self.norm_src_embedding_matrix = xp.asarray(self.norm_src_embedding_matrix)
        self.norm_trg_embedding_matrix = xp.asarray(self.norm_trg_embedding_matrix)

        # Build the seed dictionary
        # Number Vocab for induction of initial dictionary
        sim_size = min(self.norm_src_embedding_matrix.shape[0], self.norm_trg_embedding_matrix.shape[0],
                       self.unsupervised_vocab)

        # Align Embeddings
        u, s, vt = xp.linalg.svd(self.norm_src_embedding_matrix[:sim_size], full_matrices=False)
        src_sim = (xp.matmul(u, xp.diag(s))).dot(u.T)
        u, s, vt = xp.linalg.svd(self.norm_trg_embedding_matrix[:sim_size], full_matrices=False)
        trg_sim = (xp.matmul(u, xp.diag(s))).dot(u.T)
        del u, s, vt
        src_sim.sort(axis=1)
        trg_sim.sort(axis=1)
        src_sim = vecmap_normalize(src_sim)
        trg_sim = vecmap_normalize(trg_sim)

        # Apply Nearest Neighbor
        sim = src_sim.dot(trg_sim.T)
        knn_sim_fwd = topk_mean(sim, k=self.csls_neighborhood)
        knn_sim_bwd = topk_mean(sim.T, k=self.csls_neighborhood)
        sim -= knn_sim_fwd[:, xp.newaxis] / 2 + knn_sim_bwd / 2
        self.src_indices = xp.concatenate((xp.arange(sim_size), xp.array(sim.argmax(axis=0)).squeeze()))
        self.trg_indices = xp.concatenate((xp.array(sim.argmax(axis=1)).squeeze(), xp.arange(sim_size)))
        del src_sim, trg_sim, sim

    def training_loop(self, use_gpu=False):
        """Start the Training Loop of VecMap

        Args:
            use_gpu: Use GPU (recommended) or not

        Returns:

        """
        if use_gpu:
            if not supports_cupy():
                print('ERROR: Install CuPy for CUDA support')
                sys.exit(-1)
            xp = get_cupy()
        else:
            xp = np

        self.norm_src_embedding_matrix = xp.asarray(self.norm_src_embedding_matrix)
        self.norm_trg_embedding_matrix = xp.asarray(self.norm_trg_embedding_matrix)

        # Allocate memory
        xw = xp.empty_like(self.norm_src_embedding_matrix)
        zw = xp.empty_like(self.norm_trg_embedding_matrix)
        src_size = self.norm_src_embedding_matrix.shape[0] if self.vocabulary_cutoff <= 0 else min(
            self.norm_src_embedding_matrix.shape[0], self.vocabulary_cutoff)
        trg_size = self.norm_trg_embedding_matrix.shape[0] if self.vocabulary_cutoff <= 0 else min(
            self.norm_trg_embedding_matrix.shape[0], self.vocabulary_cutoff)
        simfwd = xp.empty((self.batch_size, trg_size), dtype="float32")
        simbwd = xp.empty((self.batch_size, src_size), dtype="float32")

        best_sim_forward = xp.full(src_size, -100, dtype="float32")
        src_indices_forward = xp.arange(src_size)
        trg_indices_forward = xp.zeros(src_size, dtype=int)
        best_sim_backward = xp.full(trg_size, -100, dtype="float32")
        src_indices_backward = xp.zeros(trg_size, dtype=int)
        trg_indices_backward = xp.arange(trg_size)
        knn_sim_fwd = xp.zeros(src_size, dtype="float32")
        knn_sim_bwd = xp.zeros(trg_size, dtype="float32")

        # Training loop
        best_objective = -100
        objective = -100
        it = 1
        last_improvement = 0
        keep_prob = 0.1
        stochastic_interval = 50
        stochastic_multiplier = 2

        end = False
        t = time.time()
        while True:
            # Increase the keep probability if we have not improve in args.stochastic_interval iterations
            if it - last_improvement > stochastic_interval:
                if keep_prob >= 1.0:
                    end = True
                keep_prob = min(1.0, stochastic_multiplier * keep_prob)
                last_improvement = it

            if not end:  # orthogonal mapping
                u, s, vt = xp.linalg.svd(self.norm_trg_embedding_matrix[self.trg_indices].T.dot(
                    self.norm_src_embedding_matrix[self.src_indices]))
                w = vt.T.dot(u.T)
                self.norm_src_embedding_matrix.dot(w, out=xw)
                zw[:] = self.norm_trg_embedding_matrix

            else:  # advanced mapping

                # Advanced mapping
                xw[:] = self.norm_src_embedding_matrix
                zw[:] = self.norm_trg_embedding_matrix

                # STEP 1: Whitening
                def whitening_transformation(m):
                    u, s, vt = xp.linalg.svd(m, full_matrices=False)
                    return vt.T.dot(xp.diag(1 / s)).dot(vt)

                # Whiten
                if True:
                    wx1 = whitening_transformation(xw[self.src_indices])
                    wz1 = whitening_transformation(zw[self.trg_indices])
                    xw = xw.dot(wx1)
                    zw = zw.dot(wz1)

                # STEP 2: Orthogonal mapping
                wx2, s, wz2_t = xp.linalg.svd(xw[self.src_indices].T.dot(zw[self.trg_indices]))
                wz2 = wz2_t.T
                xw = xw.dot(wx2)
                zw = zw.dot(wz2)

                # STEP 3: Re-weighting
                xw = xp.matmul(xw, xp.diag(xp.power(s, self.src_reweight)))
                zw = xp.matmul(zw, xp.diag(xp.power(s, self.trg_reweight)))

                # STEP 4: De-whitening
                # Source
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
                # Target
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

                # STEP 5: Dimensionality reduction
                if self.dim_reduction > 0:
                    xw = xw[:, :self.dim_reduction]
                    zw = zw[:, :self.dim_reduction]

            # Self-learning
            if end:
                break
            else:
                # Update the training dictionary - Forward
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                    knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=self.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                    simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j - i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])

                # Update the training dictionary - Backwards
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j - i])
                    knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=self.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j - i])
                    simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j - i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])

                self.src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                self.trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))
                # Objective function evaluation
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2

                if objective - best_objective >= self.threshold:
                    last_improvement = it
                    best_objective = objective

                if it % 10 == 0:
                    duration = time.time() - t
                    print('ITERATION {0} ({1:.2f}s)'.format(it, duration))
                    print('\t- Objective:        {0:9.4f}%'.format(100 * objective))
                    print('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob))
                    t = time.time()

            
            it += 1
        if use_gpu:
            self.proj_embedding_source_target = xp.asnumpy(xw)
            self.norm_trg_embedding_matrix = xp.asnumpy(zw)
        else:
            self.proj_embedding_source_target = xw
            self.norm_trg_embedding_matrix = zw

    def create_cross_lingual_word_embedding(self, use_gpu=False):
        """Induce the CLWE with VecMap

        Args:
            use_gpu: Use GPU (recommended) or not

        Returns:

        """
        s = time.time()
        self.build_seed_dictionary(use_gpu)
        self.training_loop(use_gpu)
        print("Total Training Time: {} minutes".format((time.time() - s) / 60))

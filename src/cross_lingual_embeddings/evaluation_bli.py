import numpy as np
from src.cross_lingual_embeddings.utils import normalize_matrix
from src.cross_lingual_embeddings.load_monolingual import load_translation_dict


class Evaluator():
    # Test Translation Dictionary
    test_translation_source = []
    test_translation_target = []

    CrossLingualModel = None

    def __init__(self, CrossLingualModel, test_translation_dict_path):

        # Built Test Translation Dictionary
        self.test_translation_source, self.test_translation_target = load_translation_dict(test_translation_dict_path)

        # Select CrossLingualModel to test
        self.CrossLingualModel = CrossLingualModel

    def evaluation_on_BLI(self):
        ranking = []
        iteration = 0
        norm_proj_src_emb = normalize_matrix(self.CrossLingualModel.proj_embedding_source_target)
        for test_src_word, test_trg_word in zip(self.test_translation_source, self.test_translation_target):

            source_index = self.CrossLingualModel.src_word2ind[
                test_src_word] if test_src_word in self.CrossLingualModel.src_word2ind.keys() else -1
            target_index = self.CrossLingualModel.trg_word2ind[
                test_trg_word] if test_trg_word in self.CrossLingualModel.trg_word2ind.keys() else -1
            if source_index == -1 or target_index == -1:
                continue

            # Calculate Cos Similarity
            norm_proj_src_word_emb = norm_proj_src_emb[[source_index]]
            similarity_cos = np.dot(norm_proj_src_word_emb,
                                    np.transpose(self.CrossLingualModel.norm_trg_embedding_matrix))
            # Find Closest Neighbors
            most_similar_trg_index = np.argsort(-similarity_cos[[0]])
            find_rank = np.where(most_similar_trg_index == target_index)[1][0] + 1
            ranking.append(find_rank)

            if iteration <= 5:
                print("\nTest translation: {} -> {}".format(test_src_word,
                                                            self.CrossLingualModel.trg_ind2word[target_index]))
                print("Predicted Top 3 Translations: {}, {}, {}".format(
                    self.CrossLingualModel.trg_ind2word[most_similar_trg_index[0, 0]],
                    self.CrossLingualModel.trg_ind2word[most_similar_trg_index[0, 1]],
                    self.CrossLingualModel.trg_ind2word[most_similar_trg_index[0, 2]]))
            iteration += 1

        if len(ranking) == 0:
            print("NO MATCHING FOUND!")
        else:
            print("\n\nNumber of Test Translations: {}/{}".format(len(ranking), len(self.test_translation_source)))
            p1 = len([p for p in ranking if p <= 1]) / len(ranking)
            p5 = len([p for p in ranking if p <= 5]) / len(ranking)
            p10 = len([p for p in ranking if p <= 10]) / len(ranking)
            print("P@1: {}".format(p1))
            print("P@5: {}".format(p5))
            print("P@10: {}".format(p10))

            mrr = sum([1.0 / p for p in ranking]) / len(ranking)
            print("\n\nMRR: {}".format(mrr))

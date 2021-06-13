""" Class to use text encoders.
"""

import os
import sys

import numpy as np
import torch
import tqdm.notebook as tq
from transformers import XLMRobertaModel, XLMRobertaTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import normalize_matrix
from load_monolingual import load_translation_dict


class TextEncoders:
    """Create Text Encoder (only XLM-R for now) to create sentence embeddings.

       Attributes:
           target_embedding_matrix (array): Original Monolingual Source Embedding Matrix.
           src_word2ind (dict): Dictionary of source word to index.
           trg_word2ind (dict): Dictionary of target word to index.
           src_ind2word (dict): Dictionary of index to word source.
           trg_ind2word (dict): Dictionary of index to word target.
           norm_trg_embedding_matrix (array): Original Normalized Monolingual Source Embedding Matrix.
           proj_embedding_source_target (array): Projected Source Embedding to Target Space (CLWE).

    """
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

    batch_size = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def __init__(self, choose_model):
        """Initialize Text Encoder.

        Args:
            choose_model (str): Only XLM-R possible for now.
        """
        self.model_max_length = 128
        if choose_model.lower() == "XLM-R".lower():
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',
                                                                 model_max_length=self.model_max_length)
            self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', output_hidden_states=True)
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model.to(self.device, non_blocking=True)

            self.PAD_TOKEN = "<pad>"
            self.BOS_TOKEN = "<s>"
            self.EOS_TOKEN = "</s>"
            self.UNK_TOKEN = "<unk>"

            self.add_special_token = True
            self.pad_to_max_length = True

            self.target_embedding_matrix = []
            self.proj_embedding_source_target = []
            self.src_word2ind = {}
            self.trg_word2ind = {}
            self.src_ind2word = {}
            self.trg_ind2word = {}
            self.norm_trg_embedding_matrix = []

        else:
            assert False, print("No correct model was chosen!!")

    def tokenize(self, sequence):
        """Tokenize a sequence.

        Args:
            sequence (str): Sequence of characters.

        Returns:
            tuple: Tokenized Sequence.

        """
        return self.tokenizer.tokenize(sequence)

    def encode(self, sequence):
        """Encode the Sequence with additional information about embedding terms
        to later calculate embedding for each term

        Args:
            sequence: String Sequence.

        Returns:

        """

        word_tokens = {}
        worpiece_tokens = []

        # Subtracts Start and End Token
        _max = self.model_max_length - 2

        # Tokenize Input
        tokenized_input = self.tokenize(sequence)

        # Enumerate Token
        id_embedding = []
        word_pieces = []

        # Find Embeddings for each Token

        for index_token, word_piece in enumerate(tokenized_input):
            id_embedding.append(index_token + 1)
            word_pieces.append(word_piece)
            if index_token < len(tokenized_input) - 1:
                next_word_piece = tokenized_input[index_token + 1]
                if ("▁" == next_word_piece[0] and len(next_word_piece) >= 2) or self.BOS_TOKEN in next_word_piece:
                    word_ended = True
                else:
                    word_ended = False
            else:
                word_ended = True

            is_last_wordpiece = word_ended
            if is_last_wordpiece:
                if (len(worpiece_tokens) + len(word_pieces)) < _max:
                    worpiece_tokens.extend(word_pieces)
                    word = "".join(word_pieces)
                    word_tokens[word.replace("▁", "")] = id_embedding
                    word_pieces = []
                    id_embedding = []
                else:
                    break

        # Construct ID Tokens
        sent = [self.tokenizer.convert_tokens_to_ids(token) for token in worpiece_tokens]
        if self.add_special_token:
            sent = [self.tokenizer.convert_tokens_to_ids(self.BOS_TOKEN)] + sent + [
                self.tokenizer.convert_tokens_to_ids(self.EOS_TOKEN)]

        sequence_length = torch.tensor(len(sent))

        # Add Padding Tokens
        if self.pad_to_max_length and len(sent) < self.model_max_length:
            difference = self.model_max_length - len(sent)
            sent = sent + [self.tokenizer.convert_tokens_to_ids(self.PAD_TOKEN)] * difference
        sent = torch.LongTensor(sent).cuda()

        return sent, sequence_length, word_tokens

    def embedding(self, sequence):
        """ Create Embedding for sequence.

        Args:
            sequence: str, Text Input

        Returns:
            list: Embedding for sequence
            list: Word Tokens

        """

        sent, _, word_tokens = self.encode(sequence)
        outputs = self.model(input_ids=sent.unsqueeze(0), return_dict=True)
        del sent

        return outputs, word_tokens

    def create_embedding_for_each_term(self, sequence, use_layer=11):
        """ Create Embedding for each term by taking the first subtoken embedding.

        Args:
            sequence: str, Text Input
            use_layer: Layer to take embeddings from.

        Returns:
            array: Embedding for each term
            dict: Dictionary term to id
            dict: Dictionary id to term

        """
        outputs, word_tokens = self.embedding(sequence)
        layer_output = outputs[2][use_layer].squeeze()

        embedding_each_term = []
        term2id = {}
        id2term = {}
        # Create Embedding for each term
        for index, (word, index_embedding_lst) in enumerate(word_tokens.items()):
            # Take the first subtoken embedding of term to represent
            embedding_each_term.append(layer_output[index_embedding_lst[0]].cpu().detach().numpy())
            term2id[word] = index
            id2term[index] = word

        embedding_each_term = np.array(embedding_each_term)
        del outputs
        return embedding_each_term, term2id, id2term

    def create_source_target_embedding(self, test_translation_dict_path, use_layer=11):
        """Create Embeddings for each word in the given dictonary (single words).

        Args:
            test_translation_dict_path: path to dictionary
            use_layer: Layer to take embeddings from

        Returns:

        """

        # Load Dictionary
        source_word_translation, target_word_translation = load_translation_dict(test_translation_dict_path)

        for source_index, source_word in tq.tqdm(enumerate(source_word_translation),
                                                 total=len(source_word_translation)):
            # Word to index map
            self.src_word2ind[source_word] = source_index
            self.src_ind2word[source_index] = source_word
            embedding_each_term, _, _ = self.create_embedding_for_each_term(source_word, use_layer=use_layer)
            self.proj_embedding_source_target.append(embedding_each_term.squeeze())
            del embedding_each_term
            torch.cuda.empty_cache()

        for target_index, target_word in tq.tqdm(enumerate(target_word_translation),
                                                 total=len(target_word_translation)):
            # Word to index map
            self.trg_word2ind[target_word] = target_index
            self.trg_ind2word[target_index] = target_word
            embedding_each_term, _, _ = self.create_embedding_for_each_term(target_word, use_layer=use_layer)
            self.target_embedding_matrix.append(embedding_each_term.squeeze())
            del embedding_each_term
            torch.cuda.empty_cache()

        self.proj_embedding_source_target = np.array(self.proj_embedding_source_target)
        self.target_embedding_matrix = np.array(self.target_embedding_matrix)
        self.norm_trg_embedding_matrix = normalize_matrix(self.target_embedding_matrix)

    def create_embedding_sentence(self, sentence):
        """Create Embedding for each term in the sentence.

        Args:
            sentence: str, Text Input

        Returns:
            array: Embedding for each term
            dict: Dictionary term to id
            dict: Dictionary id to term

        """
        embedding_each_term, term2id, id2term = self.create_embedding_for_each_term(sentence, use_layer=12)
        return embedding_each_term, term2id, id2term

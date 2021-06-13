""" Functions to preprocess parallel sentence data.
"""

import pickle
import re
import string

import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.utils.timer import timer

tqdm.pandas()


@timer
def tokenize_sentence(sentence_vector):
    """ Function to tokenize an array of sentences.
       Args:
           sentence_vector (numpy.array): Array containing text.
       Returns:
           numpy.array: Array containing the individual tokens of the input sentence.
    """
    return sentence_vector.progress_apply(lambda sentence: word_tokenize(sentence))


@timer
def strip_whitespace(token_vector):
    """ Function to strip whitespaces of an array of sentences.
        Args:
            token_vector (numpy.array): Array containing text.
        Returns:
            numpy.array: Array containing the individual tokens of the input sentence without possible whitespaces.
    """
    return token_vector.progress_apply(lambda word: list(map(str.strip, word)))


@timer
def spacy(sentence_vector, nlp_language):
    """ Function to lemmatize an array of sentences.

        Args:
            sentence_vector (array): Array containing text.
            nlp_language (object): Spacy pipeline.

        Returns:
            numpy.array: Array containing the lemmatized words.
    """
    lemmatizer_language = nlp_language.get_pipe("lemmatizer")
    return sentence_vector.progress_apply(lambda sentence: [token for token in nlp_language(sentence)])


@timer
def remove_stopwords(token_vector, stopwords_list):
    """ Function to remove stopwords out of an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text.
            stopwords_list (list): List of stopwords in a specific language.

        Returns:
            numpy.array: Array containing tokenized sentence removed stopwords.
    """

    return token_vector.progress_apply(lambda token_list: [word for word in token_list if word not in stopwords_list])


@timer
def lemmatize(sentence_vector):
    """ Function to lemmatize an array of sentences.

        Args:
            sentence_vector (array): Array containing text.

        Returns:
            numpy.array: Array containing the lemmatized words.
    """
    return sentence_vector.progress_apply(lambda token_list:
                                          [token.lemma_ for token in token_list])


@timer
def lowercase(token_vector):
    """ Function to lowercase an array of sentences.
        Args:
            token_vector (numpy.array): Array containing tokenized sentence.
        Returns:
            numpy.array: Array containing tokenized, lowercased sentence.
    """
    return token_vector.progress_apply(lambda row: list(map(str.lower, row)))


@timer
def lowercase_spacy(token_vector):
    """ Function to lowercase an array of sentences.

        Args:
            token_vector (numpy.array): Array containing tokenized sentence.

        Returns:
            numpy.array: Array containing tokenized, lowercased sentence.
    """
    return token_vector.progress_apply(lambda token_list: [token.lower() for token in token_list])


@timer
def remove_punctuation(token_vector):
    """ Function to remove punctuation out of an array of sentences.

        Args:
            token_vector (numpy.array): Array containing tokenized, lowercased sentence.

        Returns:
            numpy.array: Array containing tokenized sentence removed punctuation.
    """
    return token_vector.progress_apply(lambda token_list: [word for word in token_list if not word.is_punct])


@timer
def remove_numbers(token_vector):
    """ Function to remove numbers out of an array of sentences.

        Args:
            token_vector (numpy.array): Array containing tokenized, lowercased sentence.

        Returns:
            numpy.array: Array containing tokenized sentence removed numbers.
    """
    return token_vector.progress_apply(lambda token_list: [word for word in token_list if not word.like_num])


@timer
def create_cleaned_token_embedding(sentence_vector, nlp_language, stopwords_list):
    """ Function combine cleaning function for embedding-based features.

    Args:
        sentence_vector (numpy.array): Array containing text.
        nlp_language (object): Spacy pipeline.
        stopwords_list (list) : List of stopwords.

    Returns:
        array: Cleaned array as Bag of Words.
    """
    token_vector_spacy = spacy(sentence_vector, nlp_language)
    token_vector_punctuation = remove_punctuation(token_vector_spacy)
    token_vector_numbers = remove_numbers(token_vector_punctuation)
    sentence_vector_lemmatized = lemmatize(token_vector_numbers)
    token_vector_preprocessed = lowercase_spacy(sentence_vector_lemmatized)
    token_vector_stopwords = remove_stopwords(token_vector_preprocessed, stopwords_list)

    return token_vector_stopwords


@timer
def create_cleaned_text(sentence_vector, stopwords_list):
    """ Function combine cleaning function for text-based features.

    Args:
        sentence_vector (array): Array containing text.
        stopwords_list (list): List of stopwords.

    Returns:
        array: Cleaned array as Bag of Word.
    """
    token_vector = tokenize_sentence(sentence_vector)
    token_vector_stopwords = remove_stopwords(token_vector, stopwords_list)
    token_vector_whitespace = strip_whitespace(token_vector_stopwords)
    token_vector_lowercase = lowercase(token_vector_whitespace)
    token_vector_stopwords = remove_stopwords(token_vector_lowercase, stopwords_list)

    return token_vector_stopwords


@timer
def number_punctuations_total(sentence_vector):
    """ Function to generate the number of all punctuation marks in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (array): Bag of Words array.

       Returns:
           array: Array containing the total number of punctuation marks.
    """
    list_pm = list(string.punctuation)

    # Drop the end of sentence points, since it is not an differentiator between two sentences. And the data set may
    # translate two sentences or more into one.
    list_pm.remove('.')
    list_pm.append('...')

    return sentence_vector.progress_apply(lambda sentence: len([word for word in sentence if word in list_pm]))


@timer
def number_words(sentence_vector):
    """ Function to generate the number of words in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (array): Bag of Words array.

       Returns:
           array: Array containing the total number of words.
       """
    return sentence_vector.progress_apply(
        lambda sentence: len([word for word in sentence if word not in string.punctuation]))


@timer
def number_unique_words(sentence_vector):
    """ Function to generate the number of unique words in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (array): Bag of Words array.

       Returns:
           array: Array containing the total number of unique words.
       """
    return sentence_vector.progress_apply(
        lambda sentence: len(np.unique([word for word in sentence if word not in string.punctuation])))


@timer
def number_punctuation_marks(sentence_vector, punctuation_mark):
    """ Function to generate the number of a given punctuation mark in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (array): Bag of Words array.
           punctuation_mark (str): Punctuation mark of interest.

       Returns:
           array: Array containing the total number of this punctuation mark.
       """
    return sentence_vector.progress_apply(lambda sentence: len([word for word in sentence if word == punctuation_mark]))


@timer
def number_characters(sentence_vector):
    """ Function to generate the number of characters in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (array): Bag of Words array.

       Returns:
           array: Array containing the total number of characters.
       """
    return sentence_vector.progress_apply(lambda sentence:
                                          np.sum([len(word) for word in sentence if word not in string.punctuation]))


@timer
def average_characters(character_vector, word_vector):
    """ Function to generate the number of characters per word in a given vector of Bag of Words-Sentences.

       Args:
           character_vector (array): array containing the amount of characters.
           word_vector (array): array containing the amount of words.

       Returns:
           numpy.array: Array containing the average amount of characters per word.
       """
    return (character_vector / word_vector).replace(np.nan, 0).replace(np.inf, 0).replace(np.log(0), 0)


@timer
def number_pos(sentence_vector, pos):
    """ Function to generate the number of a given part-of-speech tag in a given vector of Bag of Words-Sentences.

       Args:
           sentence_vector (numpy.array): Bag of Words array.
           pos: a given part-of-speech tag.

       Returns:
           numpy.array: Array containing the total number of a given part-of-speech tag.
       """

    return sentence_vector.progress_apply(
        lambda sentence: len([token for token in sentence if token.pos_ == pos]))


@timer
def number_times(sentence_vector, tense):
    """ Function to generate the number of a given tense verb tag in a given vector of Bag of Words-Sentences.

    Args:
           sentence_vector (numpy.array): Bag of Words array.
           tense: a given verb tense tag.

    Returns:
           numpy.array: Array containing the total number of verbs in a given tense.

    """
    return sentence_vector.progress_apply(
        lambda sentence: len([token for token in sentence if token.morph.get(
            "Tense") == tense]))


# @timer
# def polarity(sentence_vector, textblob_language):
#     """ Function to generate the polarity in a given vector of Bag of Words-sentences.
#
#        Args:
#            sentence_vector (array): Bag of Words array.
#            textblob_language (str): Language of the array.
#
#        Returns:
#            array: Array containing the polarity (sentiment analyses).
#        """
#     return sentence_vector.progress_apply(lambda sentence: textblob_language(sentence).sentiment.polarity)
#
#
# @timer
# def subjectivity(sentence_vector, textblob_language):
#     """ Function to generate the subjectivity in a given vector of Bag of Words-sentences.
#
#         Args:
#             sentence_vector (array): Bag of Words array.
#             textblob_language (str): Language of the array.
#
#         Returns:
#             array: Array containing the subjectivity (sentiment analyses).
#
#         """
#     return sentence_vector.progress_apply(lambda sentence: textblob_language(sentence).sentiment.subjectivity)


@timer
def number_stopwords(sentence_vector, stopwords_language):
    """ Function to generate the number of stopwords in a given vector of Bag of Words-Sentences.

        Args:
            sentence_vector (array): Bag of Words array.
            stopwords_language (str): Stopwords in the language of the array.

        Returns:
            array: Array containing the total number of stopwords in a given language.

        """
    return sentence_vector.progress_apply(
        lambda sentence: len([word for word in sentence if word in stopwords_language]))


# def named_entities(sentence_vector, nlp_language):
#     """ Function to generate the subjectivity in a given vector of Bag of Words-sentences.
#
#         Args:
#             sentence_vector (array): Bag of Words array.
#             nlp_language (str): Language of the array.
#
#         Returns:
#             array: Array containing the total number of named entities in a given language.
#
#         """
#     return sentence_vector.progress_apply(
#         lambda sentence: [name for name in nlp_language(sentence).ents])

@timer
def named_numbers(token_vector):
    """ Function to remove numbers out of an array of sentences.

        Args:
            token_vector (array): Array containing tokenized, lowercased sentence.

        Returns:
            array: Array containing tokenized sentence removed numbers.

        """
    return token_vector.progress_apply(lambda sentence: re.findall(r'\d+', sentence))


@timer
def load_embeddings(embedding_array_path,
                    embedding_dictionary_path):
    """ Function to load embeddings.

    Args:
        embedding_array_path (string): Path to the array of embeddings.
        embedding_dictionary_path (string): Path to the dictionary matching words to embeddings.

    Returns:
        array: Array containing normalized embeddings
        dictionary: Dictionary matching words to embeddings.
    """
    with open(embedding_array_path, 'rb') as fp:
        embedding_array = pickle.load(fp)
    with open(embedding_dictionary_path, 'rb') as fp:
        embedding_dictionary = pickle.load(fp)

    def normalize_array(array):
        """ Function to normalize embeddings.
        """
        norms = np.sqrt(np.sum(np.square(array), axis=1))
        norms[norms == 0] = 1
        norms = norms.reshape(-1, 1)
        array /= norms[:]
        return array

    embedding_array_normalized = normalize_array(np.vstack(embedding_array))

    return embedding_array_normalized, embedding_dictionary


@timer
def pca_embeddings(embedding_array_normalized, k=10):
    """ Calculate Principal Components of normalized mebeddings.

    Args:
        embedding_array_normalized (array): Normalized embedding array.
        k (int): Number of principal components

    Returns:
        array: Array containing principal components of the embeddings
    """
    pca = PCA(n_components=k)
    principal_components = pca.fit_transform(embedding_array_normalized)
    return np.asarray(principal_components)


@timer
def word_embeddings(token_vector, embedding_array, embedding_dictionary):
    """ Function to create embeddings for the preprocessed words.

       Args:
           token_vector (array): Array containing text.
           embedding_array (array): Array containing embeddings.
           embedding_dictionary (dictionary): Dictionary matching words to embeddings.

       Returns:
           pandas.Dataframe: Array containing arrays of the embeddings.
       """

    def token_list_embedding(embedding_array, embedding_dictionary, token_list):
        """ Function to retrieve the embeddings from the array.
        """
        word_embedding_dictionary = {}
        for i in range(len(token_list)):
            if embedding_dictionary.get(token_list[i]):
                word_embedding_dictionary[token_list[i]] = embedding_array[
                    embedding_dictionary.get(token_list[i])].tolist()
        embedding_dataframe = pd.DataFrame(word_embedding_dictionary)
        return embedding_dataframe

    return token_vector.progress_apply(lambda token_list: token_list_embedding(embedding_array, embedding_dictionary,
                                                                               token_list))


@timer
def create_translation_dictionary(token_vector_source, token_vector_target,
                                  embedding_array_normalized_source, embedding_dictionary_source,
                                  embedding_array_normalized_target, embedding_dictionary_target):
    """ Function to translate words based on crosslingual word embeddings.

    Args:
        token_vector_source (array): Array containg preprocessed token in source language.
        token_vector_target (array): Array containg preprocessed token in target language.
        embedding_array_normalized_source (array): Normalized embeddings from source language.
        embedding_dictionary_source (array): Dictionary matching words to embedding in the source language.
        embedding_array_normalized_target (dict): Normalized embeddings from target language.
        embedding_dictionary_target (dict): Dictionary matching words to embedding in the target language.

    Returns:
          dict: Translation Dictionary form source to target.
          dict: Translation Dictionary form target to source.
    """
    unique_token_source = set([item for sublist in token_vector_source for item in sublist])
    unique_token_target = set([item for sublist in token_vector_target for item in sublist])

    def create_dictionary(unique_token, embedding_dictionary, embedding_array_normalized):
        """ Create reduced dictionary and embedding array for translation search.
        """
        index = 0
        word_embedding_dictionary = {}
        embedding_subset_dictionary = {}
        for token in unique_token:
            if embedding_dictionary.get(token):
                word_embedding_dictionary[token] = embedding_array_normalized[
                    embedding_dictionary.get(token)].tolist()
                embedding_subset_dictionary[index] = token
                index += 1
        return word_embedding_dictionary, embedding_subset_dictionary

    word_embedding_dictionary_source, embedding_subset_dictionary_source = create_dictionary(unique_token_source,
                                                                                             embedding_dictionary_source,
                                                                                             embedding_array_normalized_source)

    word_embedding_dictionary_target, embedding_subset_dictionary_target = create_dictionary(unique_token_target,
                                                                                             embedding_dictionary_target,
                                                                                             embedding_array_normalized_target)

    embedding_subset_source = np.array(list(word_embedding_dictionary_source.values()))
    embedding_subset_target = np.array(list(word_embedding_dictionary_target.values()))

    def translation(token, word_embedding_dictionary_source, embedding_subset_target,
                    embedding_subset_dictionary_target):
        """ Find translations in the other language and return.
        """
        norm_src_word_emb = word_embedding_dictionary_source[token]
        similarity_cos = np.dot(norm_src_word_emb, np.transpose(embedding_subset_target))
        most_similar_trg_index = np.argsort(-similarity_cos)[0].tolist()
        return embedding_subset_dictionary_target[most_similar_trg_index]

    translation_to_target_source = {}
    for token in unique_token_source:
        if embedding_dictionary_source.get(token):
            translation_to_target_source[token] = translation(token, word_embedding_dictionary_source,
                                                              embedding_subset_target,
                                                              embedding_subset_dictionary_target)

    translation_to_source_target = {}
    for token in unique_token_target:
        if embedding_dictionary_target.get(token):
            translation_to_source_target[token] = translation(token, word_embedding_dictionary_target,
                                                              embedding_subset_source,
                                                              embedding_subset_dictionary_source)

    return translation_to_target_source, translation_to_source_target


@timer
def translate_words(token_vector, translation_dictionary):
    """ Translate words based on a translation dictionary.

    Args:
        token_vector (array): Array of lists of preprocessed tokens.
        translation_dictionary (dict): Dictionary of translations.

    Returns:
        array: Translated sentence token.
    """

    def calculate_translations(word_list, translation_dictionary):
        """ Calculate translations of a list of token based on the dictionary.
        """
        translation_list = []
        for word in word_list:
            try:
                translation_list.append(translation_dictionary[word])
            except KeyError:
                continue
        return translation_list

    return token_vector.progress_apply(lambda token_list: calculate_translations(token_list, translation_dictionary))


@timer
def sentence_embedding_average(embedding_token_vector):
    """ Function to create average sentence embedding.

       Args:
           embedding_token_vector (array): Array containing embedding array of the token.

       Returns:
           array: Array containing average of the embeddings.
    """
    return embedding_token_vector.progress_apply(lambda embedding_token: [embedding_token.mean(axis=1)])


@timer
def tf_idf_vector(token_vector):
    """ Create tf-idf weightig dictionaries.
    Args:
        token_vector (array): Array containing preprocessed token vectors.

    Returns:
        array: array of dictionaries for td-idf weigths.
    """

    def identity_tokenizer(text):
        """ Identiy function.
        """
        return text

    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    tf_idf_matrix = vectorizer.fit_transform(token_vector)
    token_names = vectorizer.get_feature_names()
    tf_idf_dict = {}
    j = 0
    for name in token_names:
        tf_idf_dict[name] = j
        j += 1

    tf_idf_list = []
    for i in tqdm(range(len(token_vector))):
        tf_idf_token = {}
        for token in token_vector[i]:
            tf_idf_token[token] = tf_idf_matrix[i, tf_idf_dict[token]]
        tf_idf_list.append(tf_idf_token)
    tf_idf_vec = np.array(tf_idf_list)
    return tf_idf_vec


@timer
def sentence_embedding_tf_idf(embedding_array_dataframe_vec, tf_idf_dict_vec):
    """ Function to create tf-idf weighted sentence embeddings.

        Args:
            embedding_array_dataframe_vec (array): Array containing dataframes containing embeddings for each word in a
                                               sentence.
            tf_idf_dict_vec (dictionary): Path to the embedding dictionary.

        Returns:
            array: Array containing a tf-idf weighted sentence embedding.
        """
    df = pd.DataFrame()
    df['embedding_array_vec'] = embedding_array_dataframe_vec
    df['tf_idf_dict_vec'] = tf_idf_dict_vec

    def aggregate_tf_idf_embedding(embedding_dataframe, tf_idf_dict):
        """ Calculate tf-idf mean of word embeddings for one sentence.
        """
        for i in range(embedding_dataframe.shape[1]):
            embedding_dataframe[embedding_dataframe.columns[i]] = embedding_dataframe[embedding_dataframe.columns[i]] * \
                                                                  tf_idf_dict[embedding_dataframe.columns[i]]
        return [pd.Series(embedding_dataframe.values.mean(axis=1))]

    weighted_average = df.progress_apply(lambda x: aggregate_tf_idf_embedding(x.embedding_array_vec, x.tf_idf_dict_vec),
                                         axis=1)

    return weighted_average

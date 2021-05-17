""" Functions to implement possible preprocessing steps:
- Tokenization
- Removing white spaces
- Transform to lower case
- Removal of Punctuations
- Removal of Stopwords
- Lemmatization
- Total number of punctuation marks
- Count of different punctuation marks
- Number of words
- Number of unique words
- Number characters
- Number of different POS tags
- Number of different times
- Number of different Named Entities
- Number of Stopwords
- Sentiment Analysis

TODO: Add Translation of Words

Possible next steps:
- Apply spelling correction
- Expanding Contractions
- Converting numbers into words or removing numbers
- Remove special characters
- Expanding abbreviations

"""
import pickle
import string

import numpy as np
from nltk.tokenize import word_tokenize


def lemmatize(sentence_vector, nlp_language):
    """ Function to lemmatize an array of tokens.

        Args:
            sentence_vector (numpy.array): Array containing text
            nlp_language

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    lemmatizer_language = nlp_language.get_pipe("lemmatizer")
    return sentence_vector.apply(lambda sentence: " ".join([token.lemma_ for token in nlp_language(sentence)]))


def tokenize_sentence(sentence_vector):
    """ Function to tokenize an array of sentences.

       Args:
           sentence_vector (numpy.array): Array containing text

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    return sentence_vector.apply(lambda sentence: word_tokenize(sentence))


def strip_whitespace(token_vector):
    """ Function to strip whitespaces of an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return token_vector.apply(lambda word: list(map(str.strip, word)))


def lowercase(token_vector):
    """ Function to lowercase an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return token_vector.apply(lambda row: list(map(str.lower, row)))


def remove_punctuation(token_vector):
    """ Function to lowercase an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return token_vector.apply(lambda sentence: [word for word in sentence if word not in string.punctuation])


def remove_stopwords(token_vector, stopwords_language):
    """ Function to lowercase an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text
            stopwords_language (list): List of stopwords in a specific language

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return token_vector.apply(lambda sentence: [word for word in sentence if word not in stopwords_language])


def create_cleaned_token(sentence_vector, nlp_language, stopwords_language):
    """ Function to lemmatize an array of tokens.

    Args:
        sentence_vector (numpy.array): Array containing text
        nlp_language
        stopwords_language

    Returns:
        numpy.array: Array containing the total number of punctuation marks

    """
    sentence_vector_lemmatized = lemmatize(sentence_vector, nlp_language)
    token_vector = tokenize_sentence(sentence_vector_lemmatized)
    token_vector_whitespace = strip_whitespace(token_vector)
    token_vector_lowercase = lowercase(token_vector_whitespace)
    token_vector_punctuation = remove_punctuation(token_vector_lowercase)
    token_vector_preprocessed = remove_stopwords(token_vector_punctuation, stopwords_language)

    return token_vector_preprocessed


def number_punctuations_total(sentence_vector):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    list_pm = list(string.punctuation)
    # Drop the end of sentence points, since it is not an differentiator between two sentences.
    list_pm.remove('.')
    list_pm.append('...')

    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word in list_pm]))


def number_words(sentence_vector):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing textr

       Returns:
           numpy.array: Array containing the total number of words

       """
    return sentence_vector.apply(
        lambda sentence: len([word for word in sentence if word not in string.punctuation]))


def number_unique_words(sentence_vector):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text

       Returns:
           numpy.array: Array containing the total number of unique words

       """
    return sentence_vector.apply(
        lambda sentence: len(np.unique([word for word in sentence if word not in string.punctuation])))


def number_punctuation_marks(sentence_vector, punctuation_mark):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text
           punctuation_mark (str): Punctuation mark of interest

       Returns:
           numpy.array: Array containing the total number of this punctuation mark

       """
    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word == punctuation_mark]))


def number_characters(sentence_vector):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    sentence_vector.apply(
        lambda sentence: len([word for word in sentence]))


def number_pos(sentence_vector, nlp_language, pos):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text
           nlp_language (str): Language of the array
           pos

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    return sentence_vector.apply(
        lambda sentence: len([token for token in nlp_language(sentence) if token.pos_ == pos]))


def number_times(sentence_vector, nlp_language, tense):
    """ Function to create number of times.

    Args:
           sentence_vector (numpy.array): Array containing text
           nlp_language (str): Language of the array
           tense

    Returns:
           numpy.array: Array containing the total number of punctuation marks

    """
    return sentence_vector.apply(lambda sentence: len([token for token in nlp_language(sentence) if token.morph.get(
        "Tense") == tense]))


def polarity(sentence_vector, textblob_language):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

       Args:
           sentence_vector (numpy.array): Array containing text
           textblob_language (str): Language of the array

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    return sentence_vector.apply(lambda sentence: textblob_language(sentence).sentiment.polarity)


def subjectivity(sentence_vector, textblob_language):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

        Args:
            sentence_vector (numpy.array): Array containing text
            textblob_language (str): Language of the array

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return sentence_vector.apply(lambda sentence: textblob_language(sentence).sentiment.subjectivity)


def number_stopwords(sentence_vector, stopwords_language):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

        Args:
            sentence_vector (numpy.array): Array containing text
            stopwords_language (str): Language of the array

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word in stopwords_language]))


def named_entities(sentence_vector, nlp_language):
    """ Function to generate a comparison of the number of punctuation marks for two sentences.

        Args:
            sentence_vector (numpy.array): Array containing text
            nlp_language (str): Language of the array

        Returns:
            numpy.array: Array containing the total number of punctuation marks

        """
    return sentence_vector.apply(
        lambda sentence: [name for name in nlp_language(sentence).ents])


def sentence_embedding(token_vector, embedding_matrix_path, embedding_dictionary_path):
    def token_list_embedding(embedding_array, embedding_dictionary, token_list):
        embedding_sentence = np.zeros(shape=(len(token_list), 300))
        deletion_list = []
        for i in range(len(token_list)):
            if embedding_dictionary.get(token_list[i]):
                embedding_sentence[i] = embedding_array[embedding_dictionary.get(token_list[i])]
            else:
                deletion_list.append(i)
        embedding_sentence = np.delete(embedding_sentence, deletion_list, axis=0)
        embedding_sentence_mean = np.mean(embedding_sentence, axis=0)
        return embedding_sentence_mean

    with open(embedding_matrix_path, 'rb') as matrix:
        embedding_array_all = np.asarray(pickle.load(matrix))
    with open(embedding_dictionary_path, 'rb') as dictionary:
        embedding_dictionary_all = pickle.load(dictionary)

    return token_vector.apply(lambda token_list: token_list_embedding(embedding_array_all, embedding_dictionary_all,
                                                                      token_list))


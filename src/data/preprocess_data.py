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
            numpy.array: Array containing the lemmatized words

        """
    lemmatizer_language = nlp_language.get_pipe("lemmatizer")
    return sentence_vector.apply(lambda sentence: " ".join([token.lemma_ for token in nlp_language(sentence)]))


def tokenize_sentence(sentence_vector):
    """ Function to tokenize an array of sentences.

       Args:
           sentence_vector (numpy.array): Array containing text/sentences

       Returns:
           numpy.array: Array containing the individual tokens of the input sentence

       """
    return sentence_vector.apply(lambda sentence: word_tokenize(sentence))


def strip_whitespace(token_vector):
    """ Function to strip whitespaces of an array of sentences.

        Args:
            token_vector (numpy.array): Array containing text

        Returns:
            numpy.array: Array containing the individual tokens of the input sentence without possible whitespaces

        """
    return token_vector.apply(lambda word: list(map(str.strip, word)))


def lowercase(token_vector):
    """ Function to lowercase an array of sentences.

        Args:
            token_vector (numpy.array): Array containing tokenized sentence

        Returns:
            numpy.array: Array containing tokenized, lowercased sentence

        """
    return token_vector.apply(lambda row: list(map(str.lower, row)))


def remove_punctuation(token_vector):
    """ Function to remove punctuation out of an array of sentences

        Args:
            token_vector (numpy.array): Array containing tokenized, lowercased sentence

        Returns:
            numpy.array: Array containing tokenized sentence removed punctuation

        """
    return token_vector.apply(lambda sentence: [word for word in sentence if word not in string.punctuation])


def remove_stopwords(token_vector, stopwords_language):
    """ Function to remove stopwords out of an array of sentences

        Args:
            token_vector (numpy.array): Array containing text
            stopwords_language (list): List of stopwords in a specific language

        Returns:
            numpy.array: Array containing tokenized sentence removed stopwords
        """
    return token_vector.apply(lambda sentence: [word for word in sentence if word not in stopwords_language])


def create_cleaned_token(sentence_vector, nlp_language, stopwords_language):
    """ Function combine cleaning function for embedding-based features

    Args:
        sentence_vector (numpy.array): Array containing text
        nlp_language -> spacy package function to tag words
        stopwords_language

    Returns:
        numpy.array: Cleaned array as BoW

    """
    sentence_vector_lemmatized = lemmatize(sentence_vector, nlp_language)
    token_vector = tokenize_sentence(sentence_vector_lemmatized)
    token_vector_whitespace = strip_whitespace(token_vector)
    token_vector_lowercase = lowercase(token_vector_whitespace)
    token_vector_punctuation = remove_punctuation(token_vector_lowercase)
    token_vector_preprocessed = remove_stopwords(token_vector_punctuation, stopwords_language)

    return token_vector_preprocessed


def create_cleaned_text(sentence_vector):
    """ Function combine cleaning function for text-based features

    Args:
        sentence_vector (numpy.array): Array containing text


    Returns:
        numpy.array: Cleaned array as BoW

    """
    token_vector = tokenize_sentence(sentence_vector)
    token_vector_whitespace = strip_whitespace(token_vector)
    token_vector_lowercase = lowercase(token_vector_whitespace)

    return token_vector_lowercase


def number_punctuations_total(sentence_vector):
    """ Function to generate the number of all punctuation marks in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array

       Returns:
           numpy.array: Array containing the total number of punctuation marks

       """
    # get the list of punctuation marks from the string package
    list_pm = list(string.punctuation)
    # Drop the end of sentence points, since it is not an differentiator between two sentences. And the data set may
    # translate two sentences or more into one.
    list_pm.remove('.')
    list_pm.append('...')

    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word in list_pm]))


def number_words(sentence_vector):
    """ Function to generate the number of words in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array

       Returns:
           numpy.array: Array containing the total number of words

       """
    return sentence_vector.apply(
        lambda sentence: len([word for word in sentence if word not in string.punctuation]))


def number_unique_words(sentence_vector):
    """ Function to generate the number of unique words in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array

       Returns:
           numpy.array: Array containing the total number of unique words

       """
    return sentence_vector.apply(
        lambda sentence: len(np.unique([word for word in sentence if word not in string.punctuation])))


def number_punctuation_marks(sentence_vector, punctuation_mark):
    """ Function to generate the number of a given punctuation mark in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array
           punctuation_mark (str): Punctuation mark of interest

       Returns:
           numpy.array: Array containing the total number of this punctuation mark

       """
    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word == punctuation_mark]))


def number_characters(sentence_vector):
    """ Function to generate the number of characters in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array

       Returns:
           numpy.array: Array containing the total number of characters

       """
    return sentence_vector.apply(lambda sentence:
                                 np.sum([len(word) for word in sentence if word not in string.punctuation]))


def average_characters(character_vector, word_vector):
    """ Function to generate the number of characters per word in a given vector of BoW-Sentences.

       Args:
           character_vector (numpy.array): array containing the amount of characters
           word_vector (numpy.array): array containing the amount of words

       Returns:
           numpy.array: Array containing the average amount of characters per word

       """
    return character_vector/word_vector


def number_pos(sentence_vector, nlp_language, pos):
    """ Function to generate the number of a given part-of-speech tag in a given vector of BoW-Sentences.

       Args:
           sentence_vector (numpy.array): BoW array
           nlp_language (str): Language of the array
           pos: a given part-of-speech tag

       Returns:
           numpy.array: Array containing the total number of a given part-of-speech tag

       """
    return sentence_vector.apply(lambda sentence: len([token for token in nlp_language(sentence) if token.pos_ == pos]))


def number_times(sentence_vector, nlp_language, tense):
    """ Function to generate the number of a given tense verb tag in a given vector of BoW-Sentences.

    Args:
           sentence_vector (numpy.array): BoW array
           nlp_language (str): Language of the array
           tense: a given verb tense tag

    Returns:
           numpy.array: Array containing the total number of verbs in a given tense

    """
    return sentence_vector.apply(lambda sentence: len([token for token in nlp_language(sentence) if token.morph.get(
        "Tense") == tense]))


def polarity(sentence_vector, textblob_language):
    """ Function to generate the polarity in a given vector of BoW-sentences

       Args:
           sentence_vector (numpy.array): BoW array
           textblob_language (str): Language of the array

       Returns:
           numpy.array: Array containing the polarity (sentiment analyses)

       """
    return sentence_vector.apply(lambda sentence: textblob_language(sentence).sentiment.polarity)


def subjectivity(sentence_vector, textblob_language):
    """ Function to generate the subjectivity in a given vector of BoW-sentences

        Args:
            sentence_vector (numpy.array): BoW array
            textblob_language (str): Language of the array

        Returns:
            numpy.array: Array containing the subjectivity (sentiment analyses)

        """
    return sentence_vector.apply(lambda sentence: textblob_language(sentence).sentiment.subjectivity)


def number_stopwords(sentence_vector, stopwords_language):
    """ Function to generate the number of stopwords in a given vector of BoW-Sentences.

        Args:
            sentence_vector (numpy.array): BoW array
            stopwords_language (str): Stopwords in the language of the array

        Returns:
            numpy.array: Array containing the total number of stopwords in a given language

        """
    return sentence_vector.apply(lambda sentence: len([word for word in sentence if word in stopwords_language]))


def named_entities(sentence_vector, nlp_language):
    """ unction to generate the subjectivity in a given vector of BoW-sentences

        Args:
            sentence_vector (numpy.array): BoW array
            nlp_language (str): Language of the array

        Returns:
            numpy.array: Array containing the total number of named entities in a given language

        """
    return sentence_vector.apply(
        lambda sentence: [name for name in nlp_language(sentence).ents])


def word_embeddings(token_vector, embedding_matrix_path, embedding_dictionary_path):
    def token_list_embedding(embedding_array, embedding_dictionary, token_list):
        embedding_token = np.zeros(shape=(len(token_list), 300))
        deletion_list = []
        for i in range(len(token_list)):
            if embedding_dictionary.get(token_list[i]):
                embedding_token[i] = embedding_array[embedding_dictionary.get(token_list[i])]
            else:
                deletion_list.append(i)
        embedding_token = np.delete(embedding_token, deletion_list, axis=0)
        return embedding_token

    with open(embedding_matrix_path, 'rb') as matrix:
        embedding_array_all = np.asarray(pickle.load(matrix))
    with open(embedding_dictionary_path, 'rb') as dictionary:
        embedding_dictionary_all = pickle.load(dictionary)

    return token_vector.apply(lambda token_list: token_list_embedding(embedding_array_all, embedding_dictionary_all,
                                                                      token_list))

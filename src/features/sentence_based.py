""" Functions to generate the following sentence based features:
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

"""

import numpy as np


def difference_numerical(source_array, target_array):
    """ Function to generate the difference of a given feature variable

           Args:
               source_array (numpy.array): feature array describing source language
               target_array (numpy.array): feature array describing target language

           Returns:
               numpy.array: Array containing the differences

           """

    return target_array - source_array


def relative_difference_numerical(source_array, target_array):
    """ Function to generate the relative difference of a given feature variable

               Args:
                   source_array (numpy.array): feature array describing source language
                   target_array (numpy.array): feature array describing target language

               Returns:
                   numpy.array: Array containing the relative differences

               """

    return ((target_array - source_array) / source_array).replace(np.nan, 0).replace(np.inf, 0).replace(np.inf, 0)


def normalized_difference_numerical(source_array, target_array, source_sentence_length, target_sentence_length):
    """ Function to generate the normalized difference of a given feature variable

                   Args:
                       source_array (numpy.array): feature array describing source language
                       target_array (numpy.array): feature array describing target language
                       source_sentence_length (numpy.array): array describing the length of a feature sentence in source
                       target_sentence_length (numpy.array): array describing the length of a feature sentence in target

                   Returns:
                       numpy.array: Array containing the normalized differences

                   """

    return (source_array / source_sentence_length) - (target_array / target_sentence_length)

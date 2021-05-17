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
    return source_array-target_array


def relative_difference_numerical(source_array, target_array):
    return ((source_array-target_array) / source_array).replace(np.nan, 0)


def normalized_difference_numerical(source_array, target_array, source_sentence_length, target_sentence_length):
    return (target_array/source_sentence_length) - (source_array/target_sentence_length)





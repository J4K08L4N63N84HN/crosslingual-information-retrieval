""" Functions to generate features based on preprocessed sentece informations.
"""

import numpy as np

from src.utils.timer import timer


@timer
def difference_numerical(source_array, target_array):
    """ Function to generate the difference of a given feature variable.

        Args:
            source_array (numpy.array): feature array describing source language.
            target_array (numpy.array): feature array describing target language.

        Returns:
            numpy.array: Array containing the differences.
    """

    return abs(target_array - source_array).replace(np.nan, 0).replace(np.inf, 0).replace(np.log(0), 0)


@timer
def relative_difference_numerical(source_array, target_array):
    """ Function to generate the relative difference of a given feature variable.

        Args:
            source_array (numpy.array): feature array describing source language.
            target_array (numpy.array): feature array describing target language.

        Returns:
            numpy.array: Array containing the relative differences.
    """

    return (abs(target_array - source_array) /
            (source_array + target_array)).replace(np.nan, 0).replace(np.inf, 0).replace(np.log(0), 0)


@timer
def normalized_difference_numerical(source_array, target_array, source_sentence_length, target_sentence_length):
    """ Function to generate the normalized difference of a given feature variable.

        Args:
            source_array (numpy.array): feature array describing source language.
            target_array (numpy.array): feature array describing target language.
            source_sentence_length (numpy.array): array describing the length of a feature sentence in source.
            target_sentence_length (numpy.array): array describing the length of a feature sentence in target.

        Returns:
            numpy.array: Array containing the normalized differences.
    """

    return abs((source_array / source_sentence_length) - (target_array / target_sentence_length)).replace(
        np.nan, 0).replace(
        np.inf, 0).replace(
        np.log(0), 0)

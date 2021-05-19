""" Function to import parallel sentences in two languages.
"""

import pandas as pd


def load_doc(filename):
    """ Function to load doc into memory. """
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def to_sentences(doc):
    """ Function to split a loaded document into sentences. """
    return doc.strip().split('\n')


def import_data(sentence_data_source='../data/external/europarl-v7.de-en.en',
                sentences_data_target='../data/external/europarl-v7.de-en.de',
                number_datapoints=100
                ):
    """ Function to import the data and concatenate it into a dataframe.

    Args:
        sentence_data_source (numpy.array): Array containing text
        sentences_data_target
        number_datapoints (int): Size of subset

    Returns:
        dataframe: Array containing the total number of punctuation marks

    """
    doc_source = load_doc(sentence_data_source)
    sentences_source = to_sentences(doc_source)
    # load German data
    doc_target = load_doc(sentences_data_target)
    sentences_target = to_sentences(doc_target)
    # create data frame with sentences
    df = pd.DataFrame({'text_source': sentences_source, 'text_target': sentences_target}, columns=['text_source',
                                                                                                   'text_target'])
    return df.sample(number_datapoints)  # reduce number for testing code

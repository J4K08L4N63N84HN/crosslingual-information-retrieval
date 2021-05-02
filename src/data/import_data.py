import pandas as pd


def load_doc(filename):
    """ Function to load doc into memory. """
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def to_sentences(doc):
    """ Function to split a loaded document into sentences. """
    return doc.strip().split('\n')


def import_data(file_eng='../data/external/europarl-v7.de-en.en', file_ger='../data/external/europarl-v7.de-en.de'):
    """ Function to import the data and concatenate it into a dataframe. """
    # load English data
    doc_eng = load_doc(file_eng)
    sentences_eng = to_sentences(doc_eng)
    # load German data
    doc_ger = load_doc(file_ger)
    sentences_ger = to_sentences(doc_ger)
    # create data frame with sentences
    df = pd.DataFrame({'English': sentences_eng, 'German': sentences_ger}, columns=['English', 'German'])
    return df

import numpy as np
import nltk
import spacy
# import string library function
import string


def list_to_string(s):
    """ Function to convert list to string. """
    # initialize an empty string
    str1 = " "
    # return string
    return str1.join(s)


def get_pos_text(text, nlp):
    """ Function to get pos tags with spacy pending on text input and language. """
    # use language on document
    doc = nlp(text)
    test = []
    for token in doc:
        # get word and tag and put it as tuple into a list
        test.append((token.text, token.pos_))
    return test


def get_times_text(text, nlp):
    """ Function to get verb times with spacy pending on text input and language. """
    # use language on document
    doc = nlp(text)
    test = []
    for token in doc:
        # check if word is a verb, since we are only interested in verb tenses
        if token.pos_ == 'VERB':
            # get word and tense and put it as tuple into a list, tense is delivered as List of one variable,
            # so we convert it into a string to make handling more easy
            test.append((token.text, list_to_string(token.morph.get('Tense'))))
    return test


def comparative_features(df, feature_name, lang_2_feature, lang_1_base_feature, lang_2, lang_1):
    """ Function to generate comparative features for specific input features. """
    # input features are the column names of the feature to compare
    # get differences of punctuation marks absolute and relative with respect to english as base
    df[f'{feature_name}_dif'] = df[lang_2_feature] - df[lang_1_base_feature]
    df[f'{feature_name}_dif_rel'] = df[f'{feature_name}_dif'] / df[lang_1_base_feature]
    # set NaNs to zero
    df[f'{feature_name}_dif_rel'] = df[f'{feature_name}_dif_rel'].replace(np.nan, 0)
    # get a normalized difference by dividing the with whole length of preprocessed sentence to account for different
    # lengths
    df[f'{feature_name}_dif_normal'] = (df[lang_2_feature] / df[lang_2].apply(lambda x: len(x))) - (
            df[lang_1_base_feature] / df[lang_1].apply(lambda x: len(x)))
    return df


def feature_generation(df):
    """ Function to generate comparative features. """
    # get number of punctuation marks as feature, but drop the end of sentence points
    list_pm = list(string.punctuation)
    list_pm.remove('.')
    list_pm.append('...')
    df['PM_eng'] = df['English'].apply(lambda x: len([word for word in x if word in list_pm]))
    df['PM_ger'] = df['German'].apply(lambda x: len([word for word in x if word in list_pm]))
    # get differences of words absolute, relative with respect to english as base and normalized
    comparative_features(df, 'PM', 'PM_ger', 'PM_eng', 'German', 'English')

    # get number of words as feature
    df['Words_eng'] = df['English'].apply(
        lambda x: len([word for word in x if word not in string.punctuation]))
    df['Words_ger'] = df['German'].apply(
        lambda x: len([word for word in x if word not in string.punctuation]))
    # get differences of words absolute, relative with respect to english as base and normalized
    comparative_features(df, 'Words', 'Words_ger', 'Words_eng', 'German', 'English')

    # get number of unique words
    df['Words_eng_unique'] = df['English'].apply(
        lambda x: len(np.unique([word for word in x if word not in string.punctuation])))
    df['Words_ger_unique'] = df['German'].apply(
        lambda x: len(np.unique([word for word in x if word not in string.punctuation])))
    # get differences of unique words absolute, relative with respect to english as base and normalized
    comparative_features(df, 'Words_unique', 'Words_ger_unique', 'Words_eng_unique', 'German', 'English')

    # get number of all different punctuation marks without the end of sentence point
    list_pm = list(string.punctuation)
    list_pm.remove('.')
    list_pm.append('...')
    for mark in list_pm:
        df[f'eng_{mark}'] = df['English'].apply(lambda x: len([word for word in x if word == mark]))
        df[f'ger_{mark}'] = df['German'].apply(lambda x: len([word for word in x if word == mark]))
        # get differences absolute and relative with base of english
        df[f'dif_{mark}'] = df[f'ger_{mark}'] - df[f'eng_{mark}']
        # drop absolute values since we only want comparative features in our model
        df = df.drop(columns=[f'eng_{mark}', f'ger_{mark}'])

    # get number of characters in words and the average char per word
    df['char_eng'] = df['English'].apply(
        lambda x: len(str([word for word in x if word not in string.punctuation])))
    df['char_eng_avg'] = df['char_eng'] / df['Words_eng']
    df['char_ger'] = df['German'].apply(
        lambda x: len(str([word for word in x if word not in string.punctuation])))
    df['char_ger_avg'] = df['char_ger'] / df['Words_ger']
    # get differences of characters absolute, relative with respect to english as base and normalized
    comparative_features(df, 'char', 'char_ger', 'char_eng', 'German', 'English')
    # absolute difference between avg
    df['char_dif_avg'] = df['char_ger_avg'] - df['char_eng_avg']

    # use pos-tagger and get number of nouns, verbs, adjectives, tagset universal to only get the categories
    # use spacy package
    # load language before hand to save computing power
    nlp_en = spacy.load("en_core_web_sm")
    nlp_de = spacy.load("de_core_news_sm")
    df["English_POS"] = df.English_orig.apply(lambda x: get_pos_text(x, nlp_en))
    df["German_POS"] = df.German_orig.apply(lambda x: get_pos_text(x, nlp_de))
    # get english and german tags and one hot encode them
    universal_pos = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRT', 'PRON', 'PROPN',
                     'SCONJ', 'SYM', 'VERB', 'X']
    for u_tag in universal_pos:
        df[f'eng_{u_tag}'] = df['English_POS'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == u_tag)[u_tag])
        df[f'ger_{u_tag}'] = df['German_POS'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == u_tag)[u_tag])
    # get differences in tags
    for u_tag in universal_pos:
        df[f'dif_{u_tag}'] = df[f'ger_{u_tag}'] - df[f'eng_{u_tag}']
        # delete absolute columns since we only want comparative features
        del df[f'eng_{u_tag}']
        del df[f'ger_{u_tag}']

    # use spacy package to get verb times
    # we use the already loaded language packages
    df["English_times"] = df.English_orig.apply(lambda x: get_times_text(x, nlp_en))
    df["German_times"] = df.German_orig.apply(lambda x: get_times_text(x, nlp_de))
    # get the different tenses and one hot encode them -> pres, past and other
    for tense in ['Pres', 'Past', '']:
        df[f'eng_{tense}'] = df['English_times'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == tense)[tense])
        df[f'ger_{tense}'] = df['German_times'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == tense)[tense])
        # get the difference as a feature
        df[f'{tense}_dif'] = df[f'eng_{tense}'] - df[f'ger_{tense}']
        # drop columns with absolute values since we only want comparative features in our model
        df = df.drop(columns=[f'eng_{tense}', f'ger_{tense}'])
    # change column name of '' to 'other'
    df = df.rename(columns={'_dif': 'other_tense_dif'})

    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['PM_eng', 'PM_ger'])
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['Words_eng', 'Words_ger'])
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['Words_eng_unique', 'Words_ger_unique'])
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['char_eng', 'char_ger'])
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['char_eng_avg', 'char_ger_avg'])

    return df

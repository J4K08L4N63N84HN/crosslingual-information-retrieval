import numpy as np
import nltk
from HanTa import HanoverTagger as ht
# import string library function
import string


def feature_generation(df):
    """ Function to generate comparative features. """
    # get number of punctuation marks as feature, but drop the end of senctence points
    list_pm = list(string.punctuation)
    list_pm.remove('.')
    list_pm.append('...')
    df['PM_eng'] = df['English'].apply(lambda x: len([word for word in x if word in list_pm]))
    df['PM_ger'] = df['German'].apply(lambda x: len([word for word in x if word in list_pm]))
    # get differences of punctuation marks absolute and relative with respect to english as base
    df['PM_dif'] = df['PM_ger'] - df['PM_eng']
    df['PM_dif_rel'] = df['PM_dif'] / df['PM_eng']
    # set NaNs to zero
    df['PM_dif_rel'] = df['PM_dif_rel'].replace(np.nan, 0)
    # get a normalized difference by dividing the with whole length of preprocessed sentence to account for different
    # lengths
    df['PM_dif_normal'] = (df['PM_ger'] / df['German'].apply(lambda x: len(x))) - (
            df['PM_eng'] / df['English'].apply(lambda x: len(x)))

    # get number of words as feature
    df['Words_eng'] = df['English'].apply(
        lambda x: len([word for word in x if word not in string.punctuation]))
    df['Words_ger'] = df['German'].apply(
        lambda x: len([word for word in x if word not in string.punctuation]))
    # get differences of words absolute and relative with respect to english as base
    df['Words_dif'] = df['Words_ger'] - df['Words_eng']
    df['Words_dif_rel'] = df['Words_dif'] / df['Words_eng']
    # set NaNs to zero
    df['Words_dif_rel'] = df['Words_dif_rel'].replace(np.nan, 0)
    # get a normalized difference by dividing the with whole length of preprocessed sentence to account for different
    # lengths
    df['Words_dif_normal'] = (df['Words_ger'] / df['German'].apply(lambda x: len(x))) - (
            df['Words_eng'] / df['English'].apply(lambda x: len(x)))

    # get number of unique words
    df['Words_eng_unique'] = df['English'].apply(
        lambda x: len(np.unique([word for word in x if word not in string.punctuation])))
    df['Words_ger_unique'] = df['German'].apply(
        lambda x: len(np.unique([word for word in x if word not in string.punctuation])))
    # get differences of words absolute and relative with respect to english as base
    df['Words_unique_dif'] = df['Words_ger_unique'] - df['Words_eng_unique']
    df['Words_unique_dif_rel'] = df['Words_unique_dif'] / df['Words_eng_unique']
    # set NaNs to zero
    df['Words_unique_dif_rel'] = df['Words_unique_dif_rel'].replace(np.nan, 0)
    # get a normalized difference by dividing the with whole length of preprocessed sentence to account for different
    # lengths
    df['Words_unique_dif_normal'] = (df['Words_ger_unique'] / df['German'].apply(lambda x: len(x))) - (
            df['Words_eng_unique'] / df['English'].apply(lambda x: len(x)))

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
    # absolute difference between characters and relative with respect to english as base
    df['char_dif'] = df['char_ger'] - df['char_eng']
    df['char_dif_rel'] = df['char_dif'] / df['char_eng']
    # set NaNs to zero
    df['char_dif_rel'] = df['char_dif_rel'].replace(np.nan, 0)
    # absolute difference between avg and relative with respect to english as base
    df['char_dif_avg'] = df['char_ger_avg'] - df['char_eng_avg']

    # use pos-tagger and get number of nouns, verbs, adjectives, tagset universal to only get the categories
    df['English_pos'] = df.apply(lambda x: nltk.pos_tag(x['English'], tagset='universal'), axis=1)
    # pos-tagging with nltk not suppored in german yet -> use pretrained model for german as tagger
    # pretrained model for german as tagger
    # Christian Wartena (2019). A Probabilistic Morphology Model for German Lemmatization.
    # In: Proceedings of the 15th Conference on Natural Language Processing (KONVENS 2019):
    # Long Papers. Pp. 40-49, Erlangen.
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    ger_pos = []
    for i in df.itertuples():
        ger_pos.append(list(map(lambda x: tagger.analyze(x), i.German)))
    df['German_pos'] = ger_pos

    # get english tags and one hot encode them
    universal_pos = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', 'X']
    for u_tag in universal_pos:
        df[f'eng_{u_tag}'] = df['English_pos'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == u_tag)[u_tag])

    # Stuttgart, Tübingen Tagset
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.635.8431&rep=rep1&type=pdf
    # get German Tags
    tags_german = ['ADJA', 'ADJD', 'ADV', 'APPR', 'APPRART', 'APPO', 'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOUI', 'KOUS',
                   'KON', 'KOKOM', 'NN', 'NE', 'PDS', 'PDAT', 'PIS', 'PIAT', 'PIDAT', 'PPER', 'PPOSS', 'PPOSAT',
                   'PRELS', 'PRELAT', 'PRF', 'PWS', 'PWAT', 'PWAV', 'PAV', 'PTKZU', 'PTKNEG', 'PTKVZ', 'PTKANT', 'PTKA',
                   'TRUNC', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN',
                   'VMINF', 'VMPP', 'XY']

    # one hot encoding of german tags
    for u_tag in tags_german:
        df[f'ger_{u_tag}'] = df['German_pos'].apply(
            lambda row: nltk.FreqDist(tag for (word, tag) in row if tag == u_tag)[u_tag])

    # combine the different german tag categories into the universal tags
    df['ger_ADJ'] = df['ger_ADJA'] + df['ger_ADJD']
    df['ger_ADP'] = df['ger_APPR'] + df['ger_APPRART'] + df['ger_APPO'] + df['ger_APZR']
    df['ger_CONJ'] = df['ger_KOUI'] + df['ger_KOUS'] + df['ger_KON'] + df['ger_KOKOM']
    df['ger_DET'] = df['ger_ART']
    df['ger_NOUN'] = df['ger_NN'] + df['ger_NE']
    df['ger_NUM'] = df['ger_CARD']
    df['ger_PRT'] = df['ger_PTKZU'] + df['ger_PTKNEG'] + df['ger_PTKVZ'] + df['ger_PTKANT'] + df['ger_PTKA']
    df['ger_PRON'] = df['ger_PDS'] + df['ger_PDAT'] + df['ger_PIS'] + df['ger_PIAT'] + df['ger_PIDAT'] + df['ger_PPER'] \
                     + df['ger_PPOSS'] + df['ger_PPOSAT'] + df['ger_PRELS'] + df['ger_PRELAT'] + df['ger_PRF'] \
                     + df['ger_PWS'] + df['ger_PWAT'] + df['ger_PWAV'] + df['ger_PAV']
    df['ger_VERB'] = df['ger_VVFIN'] + df['ger_VVIMP'] + df['ger_VVINF'] + \
                     df['ger_VVIZU'] + df['ger_VVPP'] + df['ger_VAFIN'] + \
                     df['ger_VAIMP'] + df['ger_VAINF'] + df['ger_VAPP'] + \
                     df['ger_VMFIN'] + df['ger_VMINF'] + df['ger_VMPP']
    df['ger_X'] = df['ger_FM'] + df['ger_ITJ'] + df['ger_TRUNC'] + df[
        'ger_XY']

    # delete german tag columns without the Adverb cause the column name and the column itself do not get changed
    tags_german_1 = ['ADJA', 'ADJD', 'APPR', 'APPRART', 'APPO', 'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOUI', 'KOUS',
                     'KON', 'KOKOM', 'NN', 'NE', 'PDS', 'PDAT', 'PIS', 'PIAT', 'PIDAT', 'PPER', 'PPOSS', 'PPOSAT',
                     'PRELS', 'PRELAT', 'PRF', 'PWS', 'PWAT', 'PWAV', 'PAV', 'PTKZU', 'PTKNEG', 'PTKVZ', 'PTKANT',
                     'PTKA', 'TRUNC', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP',
                     'VMFIN', 'VMINF', 'VMPP', 'XY']
    for u_tag in tags_german_1:
        del df[f'ger_{u_tag}']

    # get differences in tags
    for u_tag in universal_pos:
        df[f'dif_{u_tag}'] = df[f'ger_{u_tag}'] - df[f'eng_{u_tag}']
        # delete absolute columns since we only want comparative features
        del df[f'eng_{u_tag}']
        del df[f'ger_{u_tag}']

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

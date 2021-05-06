import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def data_cleansing(df):
    """ Function to clean the data before feature generation. """
    # tokenize the sentences
    df['English'] = df.apply(lambda x: word_tokenize(x['English']), axis=1)
    df['German'] = df.apply(lambda x: word_tokenize(x['German'], language='german'), axis=1)

    # lower case
    df['English'] = df['English'].apply(lambda row: list(map(str.lower, row)))
    df['German'] = df['German'].apply(lambda row: list(map(str.lower, row)))

    # treat stopwords in english
    stop_eng = stopwords.words('english')
    # count stopwords in english
    df['SWords_eng'] = df['English'].apply(lambda x: len([word for word in x if word in stop_eng]))
    # treat stopwords in german
    stop_ger = stopwords.words('german')
    # count stopwords in german
    df['SWords_ger'] = df['German'].apply(lambda x: len([word for word in x if word in stop_ger]))

    # get difference between stopwords absolute, relative with respect to english as base and
    # normalized and drop the lines afterwards
    df['SWords_dif'] = df['SWords_ger'] - df['SWords_eng']
    df['SWords_dif_rel'] = df['SWords_dif'] / df['SWords_eng']
    # set NaNs to zero
    df['SWords_dif_rel'] = df['SWords_dif_rel'].replace(np.nan, 0)
    # get a normalized difference by dividing the with whole length of preprocessed sentence to account for different
    # lengths
    df['SWords_dif_normal'] = (df['SWords_ger'] / df['German'].apply(lambda x: len(x))) - (
            df['SWords_eng'] / df['English'].apply(lambda x: len(x)))
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['SWords_ger', 'SWords_eng'])

    # remove stopwords in german
    df['German'] = df['German'].apply(lambda x: [word for word in x if word not in stop_ger])
    # remove stopwords in english
    df['English'] = df['English'].apply(lambda x: [word for word in x if word not in stop_eng])

    return df

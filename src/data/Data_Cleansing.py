from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def data_cleansing(df_selected):
    # tokenize the sentences
    df_selected['English'] = df_selected.apply(lambda x: word_tokenize(x['English']), axis=1)
    df_selected['German'] = df_selected.apply(lambda x: word_tokenize(x['German'], language='german'), axis=1)
    # lower case
    df_selected['English'] = df_selected['English'].apply(lambda row: list(map(str.lower, row)))
    df_selected['German'] = df_selected['German'].apply(lambda row: list(map(str.lower, row)))
    # remove stopwords in german and english
    stop = stopwords.words('english')
    df_selected['English'] = df_selected['English'].apply(lambda x: [word for word in x if word not in stop])
    stop = stopwords.words('german')
    df_selected['German'] = df_selected['German'].apply(lambda x: [word for word in x if word not in stop])
    return df_selected


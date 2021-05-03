from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def data_cleansing(df_selected):
    # tokenize the sentences
    df_selected['English'] = df_selected.apply(lambda x: word_tokenize(x['English']), axis=1)
    df_selected['German'] = df_selected.apply(lambda x: word_tokenize(x['German'], language='german'), axis=1)
    # lower case
    df_selected['English'] = df_selected['English'].apply(lambda row: list(map(str.lower, row)))
    df_selected['German'] = df_selected['German'].apply(lambda row: list(map(str.lower, row)))
    # treat stopwords in english
    stop = stopwords.words('english')
    # count stopwords in english
    df_selected['SWords_eng'] = df_selected['English'].apply(lambda x: len([word for word in x if word in stop]))
    # remove stopwords in english
    df_selected['English'] = df_selected['English'].apply(lambda x: [word for word in x if word not in stop])
    # treat stopwords in german
    stop = stopwords.words('german')
    # count stopwords in german
    df_selected['SWords_ger'] = df_selected['German'].apply(lambda x: len([word for word in x if word in stop]))
    # remove stopwords in german
    df_selected['German'] = df_selected['German'].apply(lambda x: [word for word in x if word not in stop])
    return df_selected


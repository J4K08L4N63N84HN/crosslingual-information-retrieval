

def build_features(df):
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
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(columns=['English_times', 'German_times'])
    # drop columns with absolute values since we only want comparative features in our model
    df = df.drop(
        columns=["English_Blob", "English_Polarity", "English_Subjectivity", "German_Blob", "German_Polarity",
                 "German_Subjectivity"])

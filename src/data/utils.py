import pandas as pd


def get_sentence_pairs(original_sentence_pairs, dataset):
    """Get the original sentence pairs, based on the given dataset source and target id

    Args:
        original_sentence_pairs (pandas dataframe): dataset with the original dataset pairs
        dataset (pandas dataframe): Given dataset to look for new sentence pair

    Returns:
        pandas dataframe: dataset containing the sentence pairs

    """

    new_training_set = pd.DataFrame(columns=['source_id', 'target_id', 'text_source', 'text_target', 'Translation'])
    current_source_id = list(dataset["source_id"].to_numpy())
    current_target_id = list(dataset["target_id"].to_numpy())
    new_training_set["text_source"] = original_sentence_pairs.iloc[current_source_id,:]["text_source"].reset_index(drop=True)
    new_training_set["text_target"] = original_sentence_pairs.iloc[current_target_id,:]["text_target"].reset_index(drop=True)
    new_training_set["source_id"] = current_source_id
    new_training_set["target_id"] = current_target_id
    new_training_set['Translation'] = new_training_set.apply(lambda row : int(row['source_id'] == row['target_id']), axis = 1)

    return new_training_set

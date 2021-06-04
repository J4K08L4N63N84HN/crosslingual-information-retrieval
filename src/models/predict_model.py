import pandas as pd


def MAP_score(source_id, target_labels, prediction):
    """ Function to compute the Mean Average Precision score of a given ranking.

        Args:
            source_id (array): Array containing the source_id of our given queries.
            target_labels (array): Array containing the target labels of our query-document testset.
            prediction  (array): Array containing the confidences of our predicitons.

        Returns:
            MAP (integer): MAP score of our ranking.
    """
    # create a target dataframe with the id of query sentences, target_labels and the predicted confidence
    result = pd.DataFrame()
    result['source_id'] = source_id
    result['Translation'] = target_labels
    result = pd.concat([source_id, target_labels], axis=1)
    result['probabilities'] = [x[1] for x in prediction]
    # rank by the source_id and get the ranking for each of the queries for all the documents
    result['rank'] = result.groupby('source_id')['probabilities'].rank(method='min', ascending=False)
    # create a new dataframe with only the right translations to get their rankings
    ranks = result[result['Translation'] == 1].reset_index()
    # compute the MAP score by first summing all inverses and dividing by the amount of queries
    sum_inverse = 0
    for i in range(0, len(ranks)):
        sum_inverse += 1 / ranks['rank'][i]
    MAP = 1 / len(ranks) * sum_inverse
    return MAP

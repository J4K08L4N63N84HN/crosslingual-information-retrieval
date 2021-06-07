import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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
    result['probabilities'] = [x[1] for x in prediction]
    # rank by the source_id and get the ranking for each of the queries for all the documents
    result['rank'] = result.groupby('source_id')['probabilities'].rank(method='average', ascending=False)
    # create a new dataframe with only the right translations to get their rankings
    ranks = result[result['Translation'] == 1].reset_index()
    # compute the MAP score by first summing all inverses and dividing by the amount of queries
    sum_inverse = 0
    for i in range(0, len(ranks)):
        sum_inverse += 1 / ranks['rank'][i]
    MAP = 1 / len(ranks) * sum_inverse
    return MAP
    
    


def feature_selection(model, scaler, trainset, testset, starting_features, added_features):
    """
    Args:
            model (ML model): Initialised model to fit the data.
            scaler (ML scaler): Scaler to scale our feature into a given range.
            trainset (dataframe): Dataframe containing our training data.
            testset (dataframe): Dataframe containing our testing data.
            starting_features (array): Array containing the starting features for our first training.
            added_features (array): Array containing the features to be added for further training.

    """
    # get the first features to train (embedding features)
    target_train = trainset['Translation']
    target_test = testset['Translation']
    data_train = trainset.filter(items=starting_features)
    data_test = testset.filter(items=starting_features)
    # scale the features
    data_train[data_train.columns] = scaler.fit_transform(data_train[data_train.columns])
    data_test[data_test.columns] = scaler.transform(data_test[data_test.columns])
    # fit the model and get the initial MapScore
    modelfit = model.fit(data_train.to_numpy(), target_train.to_numpy())
    prediction = modelfit.predict_proba(data_test.to_numpy())
    MapScore = MAP_score(testset['source_id'], target_test, prediction)
    print("The initial MAP score on test set: {:.4f}".format(MapScore))
    # iterate through all other features and add them if they improve the MapScore
    for feature in added_features[::-1]:
        data_train = trainset.filter(items=starting_features)
        data_test = testset.filter(items=starting_features)
        data_train[feature] = trainset[feature].tolist()
        data_test[feature] = testset[feature].tolist()
        data_train[data_train.columns] = scaler.fit_transform(data_train[data_train.columns])
        data_test[data_test.columns] = scaler.transform(data_test[data_test.columns])
        modelfit = model.fit(data_train.to_numpy(), target_train.to_numpy())
        prediction = modelfit.predict_proba(data_test.to_numpy())
        #print("With {} added, the MAP score on test set: {:.4f}".format(feature,
        #                                                                MAP_score(testset['source_id'], target_test,
                                                                                  #prediction)))
        if MAP_score(testset['source_id'], target_test, prediction) > MapScore:
            starting_features.append(feature)
            MapScore = MAP_score(testset['source_id'], target_test, prediction)
            print("Updated MAP score on test set with new feature {}: {:.4f}".format(feature, MapScore))


def threshold_counts(s, threshold=0):
    counts = s.value_counts(normalize=True, dropna=False)
    if (counts >= threshold).any():
        return False
    return True
    

def downsample(imbalanced_data):
    y = imbalanced_data["Translation"].astype(int)
    y = np.where((y == 0), 0, 1)
    
    # Indicies of each class' observations
    i_class0 = np.where(y == 0)[0]
    i_class1 = np.where(y == 1)[0]

    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)
    print("Class 0 size: {}".format(n_class0))
    print("Class 1 size: {}".format(n_class1))

    # For every observation of class 1, randomly sample from class 0 without replacement
    i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False)
    print("After Downsampling:")
    print("Class 0 size: {}".format(len(i_class0_downsampled)))
    print("Class 1 size: {}".format(n_class1))
    index_balanced = i_class0_downsampled.tolist() + i_class1.tolist()
    index_balanced = shuffle(index_balanced, random_state=42)
    return imbalanced_data.iloc[index_balanced, :]
            

""" Functions to apply models.
"""

import glob
import itertools
import os

import numpy as np
import pandas as pd
from pickle5 import pickle
from scipy.special import softmax
from sklearn.utils import shuffle
from tqdm import tqdm


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


def pipeline_model_optimization(model, parameter_grid, scaler, trainset, testset, starting_features, added_features,
                                threshold_map_feature_selection=0.001):
    """ Funtion to combine the feature selection and model optimization into a pipeline to use.
    Args:
            model (ML model): Initialised model to fit the data.
            parameter_grid (dictionary): Dictionary grid to test different feature combinations.
            scaler (ML scaler): Scaler to scale our feature into a given range.
            trainset (dataframe): Dataframe containing our training data.
            testset (dataframe): Dataframe containing our testing data.
            starting_features (array): Array containing the starting features for our first training.
            added_features (array): Array containing the features to be added for further training.
            threshold_map_feature_selection (float): Get a threshold float number that defines how much a features has to
                                                improve our Map score to be added into the model.

    """
    print("-----------------First do Forward Selection-----------------")
    forward_selection(model, scaler, trainset, testset, starting_features, added_features,
                      threshold_map_feature_selection)

    target_train = trainset['Translation'].astype(float)
    data_train = trainset.loc[:, starting_features]
    target_test = testset['Translation'].astype(float)
    data_test = testset.loc[:, starting_features]

    data_train.loc[:, data_train.columns] = scaler.fit_transform(data_train.loc[:, data_train.columns])
    data_test.loc[:, data_test.columns] = scaler.transform(data_test.loc[:, data_test.columns])

    print("\n\n-----------------Start Hyperparameter-tuning with Grid Search-----------------")
    best_parameter_combination, best_map_score, all_parameter_combination = grid_search_hyperparameter_tuning(
        parameter_grid, model, data_train, target_train, data_test, testset)

    return starting_features, best_parameter_combination, best_map_score, all_parameter_combination


def forward_selection(model, scaler, trainset, testset, starting_features, added_features,
                      threshold_map_feature_selection):
    """Do Forward Selection given starting features and feature pool to add

    Args:
        model (ML model): Initialised model to fit the data.
        scaler (ML scaler): Scaler to scale our feature into a given range.
        trainset (dataframe): Dataframe containing our training data.
        testset (dataframe): Dataframe containing our testing data.
        starting_features (array): Array containing the starting features for our first training.
        added_features (array): Array containing the features to be added for further training.
        threshold_map_feature_selection (int): Threshold improvement on MAP for adding a feature

    """
    length_current_start_feature = len(starting_features)
    index = 1
    map_score = 0
    while True:
        print("\nCurrent Iteration through feature list: {}".format(index))
        map_score = feature_selection(model, scaler, trainset, testset, starting_features, added_features,
                                      threshold_map_feature_selection)
        if length_current_start_feature >= len(starting_features):
            break
        length_current_start_feature = len(starting_features)
        index += 1
    print("\n-----------------Result of Feature Selection-----------------")
    print("\nBest MAP Score after feature selection: {}".format(map_score))


def feature_selection(model, scaler, trainset, testset, starting_features, added_features,
                      threshold_map_feature_selection=0.0001):
    """ Function to select features using forward selection.
    Args:
            model (ML model): Initialised model to fit the data.
            scaler (ML scaler): Scaler to scale our feature into a given range.
            trainset (dataframe): Dataframe containing our training data.
            testset (dataframe): Dataframe containing our testing data.
            starting_features (array): Array containing the starting features for our first training.
            added_features (array): Array containing the features to be added for further training.
            threshold_map_feature_selection (flot): Treshold for map feature selection.

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
        # print("With {} added, the MAP score on test set: {:.4f}".format(feature,
        #                                                                MAP_score(testset['source_id'], target_test,
        # prediction)))

        if MAP_score(testset['source_id'], target_test, prediction) > MapScore + threshold_map_feature_selection:
            starting_features.append(feature)
            MapScore = MAP_score(testset['source_id'], target_test, prediction)
            print("Updated MAP score on test set with new feature {}: {:.4f}".format(feature, MapScore))
    return MapScore


def threshold_counts(s, threshold=0):
    """Minimum number of unique values in a pandas column

    Args:
        s (pandas series): Column to check
        threshold: Minimum number of unique values in percentage

    Returns:
        boolean: If threshold is reached or not

    """
    counts = s.value_counts(normalize=True, dropna=False)
    if (counts >= threshold).any():
        return False
    return True


def grid_search_hyperparameter_tuning(parameter_grid, model, data_train, target_train, data_test,
                                      original_retrieval_dataset):
    """Do Grid Search for Hyperparameter-tuning.

    Args:
        parameter_grid (dict): Hyper-Parameter Grid
        model (ML model): Initialised model to fit the data.
        data_train (dataframe): Training Set.
        target_train: (dataframe): Labels of Training Set
        data_test (dataframe): Test Set.
        original_retrieval_dataset (dataframe): Ranking Retrieval Set to be evaluated on.

    Returns:
        arrays: returns best combination, best map score and all combinations with respective map score

    """
    if len(parameter_grid) == 0:
        return None, None, None

    keys, values = zip(*parameter_grid.items())
    all_parameter_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("Number of Parameter Combinations: {}".format(len(all_parameter_combination)))
    current_best_map_score = 0
    for parameter_combination in tqdm(all_parameter_combination, desc="Hyperparameter Tuning",
                                      total=len(all_parameter_combination)):
        # print("\nCurrent Hyperpamaters: {}".format(parameter_combination))
        model.__init__(**parameter_combination)
        # fit the model and get the initial MapScore
        try:
            modelfit = model.fit(data_train.to_numpy(), target_train.to_numpy())
            prediction = modelfit.predict_proba(data_test.to_numpy())
            MapScore = MAP_score(original_retrieval_dataset['source_id'], original_retrieval_dataset["Translation"],
                                 prediction)
        except:
            print("Model failed to fit")
            MapScore = 0
        parameter_combination["MAP_score"] = MapScore
        if current_best_map_score < MapScore:
            print("\nCurrent Best Hyperpamaters: {}".format(parameter_combination))
            print("With Map Score {:.4f}".format(MapScore))
            current_best_map_score = MapScore

    best_parameter_combination_index = np.argmax([sublist["MAP_score"] for sublist in all_parameter_combination])
    best_parameter_combination = all_parameter_combination[best_parameter_combination_index]
    best_map_score = best_parameter_combination["MAP_score"]
    best_parameter_combination.pop('MAP_score', None)
    print("\n-----------------Result of Hyperparameter Tuning-----------------")
    print("\nBest Hyperamater Settting: {}".format(best_parameter_combination))
    print("With MAP Score: {:.4f}".format(best_map_score))
    return best_parameter_combination, best_map_score, all_parameter_combination


def downsample(imbalanced_data):
    """Function to downsample a dataset to its minority class.

    Args:
        imbalanced_data (dataframe): Dataset to be downsampled

    Returns:
        dataframe: Downsampled dataframe

    """
    y = imbalanced_data["Translation"].astype(int)
    y = np.where((y == 0), 0, 1)

    # Indices of each class' observations
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


def evaluate_text_encoder(prediction_path_folder, feature_retrieval):
    """ Evaluate Text Encoder with given predictions (must be done beforehand) on retrieval dataset

    Args:
        prediction_path_folder (path): Path to Predictions of a Model
        feature_retrieval (dataframe): Retrieval Dataset to be evaluated on

    Returns:
        int: Map score

    """
    total_map_score = 0
    for index, filepath in enumerate(
            sorted(glob.iglob(os.path.join(prediction_path_folder, "*")), key=os.path.getmtime)):
        start_index = int(filepath.split("_")[-2])
        end_index = int(filepath.split("_")[-1].split(".")[0])
        with open(filepath, 'rb') as handle:
            predictions = pickle.load(handle)

        logits, labels, metrics = predictions
        pred_prob = softmax(logits, axis=1)

        map_score = MAP_score(feature_retrieval["source_id"][start_index:end_index],
                              feature_retrieval["Translation"][start_index:end_index],
                              pred_prob)

        # print("MAP score from index {} to {} is: {}".format(start_index, end_index, map_score))
        total_map_score += map_score

    total_map_score /= index + 1
    print("Result: MAP Score is: {}".format(total_map_score))

    return total_map_score

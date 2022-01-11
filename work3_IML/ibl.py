import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from statistics import mean
import pandas as pd
from time import time
from sklearn.feature_selection import mutual_info_classif, r_regression
from sklearn.preprocessing import LabelEncoder

import metrics
import policies


def get_accuracy(classification, predictions):
    accuracy = sum(predictions == classification) / len(classification)
    return accuracy


def ib1_algorithm(datasets_train, datasets_test):
    classifications_per_fold = []
    times = []
    for train, test in zip(datasets_train, datasets_test):
        start = time()
        distances = euclidean_distances(train.iloc[:, :-1], test.iloc[:, :-1])
        min_dist = np.argmin(distances, axis=0)
        train_labels = np.array(train.iloc[:, -1])
        predictions = np.array(train_labels[min_dist])

        classifications_per_fold.append(get_accuracy(test.iloc[:, -1], predictions))
        end = time()
        times.append(end - start)

    print('Test accuracy:', round(mean(classifications_per_fold), 4), 'Efficiency:', round(mean(times), 4), 'sec')


def ib2_algorithm(datasets_train, datasets_test):
    # Training
    concept_descriptions = []
    for train, test in zip(datasets_train, datasets_test):
        concept_description = pd.DataFrame(train.iloc[0]).transpose()

        for x in test.iterrows():
            distances = euclidean_distances([x[1].to_list()[:-1]], concept_description.iloc[:, :-1].values.tolist())
            min_dist = np.argmin(distances)

            # classification, add incorrect
            if x[1].to_list()[-1] != concept_description.iloc[min_dist, -1]:  # check whether they have the same class
                new_row = pd.DataFrame(list(x[1])).transpose()
                new_row.columns = concept_description.columns
                concept_description = pd.concat([concept_description, new_row], axis=0,
                                                ignore_index=True)

        concept_descriptions.append(concept_description)

    # Testing
    ib1_algorithm(concept_descriptions, datasets_test)


def ib3_algorithm(datasets_train, datasets_test, lower_threshold=-30, upper_threshold=0.5):
    # Training
    concept_descriptions = []
    for train, test in zip(datasets_train, datasets_test):
        concept_description = pd.DataFrame(train.iloc[0]).transpose()
        concept_description_record = [0]

        for x in test.iterrows():
            length_cd = len(concept_description)
            distances = euclidean_distances([x[1].to_list()[:-1]], concept_description.iloc[:, :-1].values.tolist())

            # acceptable instances in CD
            concept_description_accepted = []
            indeces = []
            for idx, record in enumerate(concept_description_record):
                if record >= upper_threshold:
                    concept_description_accepted.append(concept_description.iloc[idx].to_list())  # acceptable instances
                    indeces.append(idx)  # their index
            concept_description_accepted = pd.DataFrame(concept_description_accepted)

            # choose min_dist
            if not concept_description_accepted.empty:  # if there are acceptable
                distances_accepted = euclidean_distances([x[1].to_list()[:-1]],
                                                         concept_description_accepted.iloc[:, :-1].values.tolist())
                min_dist = indeces[np.argmin(distances_accepted)]
            else:  # if there aren't acceptable
                np.random.seed(0)
                min_dist = np.random.randint(len(concept_description))

            # classification, add incorrect
            if x[1].to_list()[-1] != concept_description.iloc[min_dist, -1]:  # check whether they have the same class
                n_columns = len(concept_description.columns)
                concept_description.columns = range(n_columns)
                pd.DataFrame(list(x[1])).transpose().columns = range(n_columns)
                concept_description = pd.concat([concept_description, pd.DataFrame(list(x[1])).transpose()], axis=0,
                                                ignore_index=True)
                concept_description_record.append(0)

            # updating records
            delete = []
            for i in range(length_cd):
                if distances[0][i] <= distances[0][min_dist]:
                    if train.iloc[i, -1] == concept_description.iloc[i, -1]:
                        concept_description_record[i] += 1
                    else:
                        concept_description_record[i] -= 1
                        if concept_description_record[i] < lower_threshold:
                            delete.append(i)

            concept_description = concept_description.drop(delete).reset_index(drop=True)
            for i in sorted(delete, reverse=True):
                del concept_description_record[i]

        concept_descriptions.append(concept_description)

    # Testing
    ib1_algorithm(concept_descriptions, datasets_test)


def kibl_algorithm(datasets_train, datasets_test, similarity, k, voting):
    classifications_per_fold = []
    times = []
    for train, test in zip(datasets_train, datasets_test):
        start = time()
        if similarity == 'euclidean':
            distances = euclidean_distances(train.iloc[:, :-1], test.iloc[:, :-1])
        elif similarity == 'canberra':
            distances = metrics.canberra_distances(train.iloc[:, :-1].to_numpy(), test.iloc[:, :-1].to_numpy())
        elif similarity == 'hvdm':
            distances = metrics.hvdm(train, test)

        nearest_neighbor_ids = np.argsort(distances, axis=0)[:k]  # indexes most similar instances
        train_labels = np.array(train.iloc[:, -1])
        nearest_neighbor_labels = train_labels[nearest_neighbor_ids]  # labels most similar instance

        if voting == 'mvs':  # most voted solution
            predictions = policies.most_voted_solution(test, nearest_neighbor_labels)
        if voting == 'modpl':  # modified plurality
            predictions = policies.modified_plurality(test, nearest_neighbor_labels)
        if voting == 'borda':
            predictions = policies.borda(test, nearest_neighbor_labels, k)
        classifications_per_fold.append(get_accuracy(test.iloc[:, -1], predictions))
        end = time()
        times.append(end - start)


    print('Test accuracy:', round(mean(classifications_per_fold), 4), 'Efficiency:', round(mean(times), 4), 'sec')


def selectionkIBLAlgorithm(datasets_train, datasets_test, FSmetrics):

    i = 0
    for train, test in zip(datasets_train, datasets_test):
        X_train = train.iloc[:, :-1]  # training features
        y_train = train.iloc[:, -1]  # training labels

        X_test = test.iloc[:, :-1]  # testing features
        y_test = test.iloc[:, -1]  # testing labels

        # label encoding training/testing labels
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)  # training
        y_test = le.transform(y_test)  # testing

        # feature weights
        if FSmetrics == 'Information Gain':
            i_scores = mutual_info_classif(X_train, y_train)
        if FSmetrics == 'Correlation':
            i_scores = r_regression(X_train, y_train)
            for j in range(len(i_scores)):
                if str(i_scores[j]) == 'nan' or str(i_scores[j]) == '-inf' or str(i_scores[j]) == 'inf':
                    i_scores[j] = 0

        # calculate feature values
        X_train = X_train * i_scores
        X_test = X_test * i_scores

        # modify feature values in the datasets
        datasets_train[i].iloc[:, :-1] = X_train
        datasets_test[i].iloc[:, :-1] = X_test

        i += 1

    # best kibl_algorithm
    kibl_algorithm(datasets_train, datasets_test, 'euclidean', 3, 'borda')


def instance_based_algorithms(datasets_train, datasets_test, algorithm):
    if algorithm == 'ib1':
        return ib1_algorithm(datasets_train, datasets_test)

    elif algorithm == 'ib2':
        return ib2_algorithm(datasets_train, datasets_test)

    elif algorithm == 'ib3':
        return ib3_algorithm(datasets_train, datasets_test)

    elif algorithm == 'kibl':
        kibl_algorithm(datasets_train, datasets_test, 'hvdm', 5, 'mvs')  # mvs, modpl, borda, euclidean, canberra

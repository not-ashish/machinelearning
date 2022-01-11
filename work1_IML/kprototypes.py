import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn import metrics

import preprocessing


def get_optimal_parameters(dataset):
    if dataset == 'kropt':
        return 2, 0.1
    if dataset == 'satimage':
        return 3, 1
    if dataset == 'hypothyroid':
        return 2, 0.5


def is_numeric(column):
    for val in column:
        if type(val) == bytes:  # categorical data found
            return False
    return True


def missing_elements(arr_seq, length):
    if not arr_seq:
        return [x for x in range(length)]
    start, end = arr_seq[0], arr_seq[-1]
    return sorted(set(range(start, end + 1)).difference(arr_seq))


def get_distance_numeric(data, indeces_numeric_columns, prototypes):
    numeric_col_prototypes = prototypes[:, indeces_numeric_columns]
    numeric_col_data = data[:, indeces_numeric_columns]
    distances = euclidean_distances(numeric_col_data.astype(np.float64), numeric_col_prototypes.astype(np.float64))
    return distances


def get_distance_nominal(data, indeces_nominal_columns, prototypes):
    nominal_col_prototypes = prototypes[:, indeces_nominal_columns]
    nominal_col_data = data[:, indeces_nominal_columns]

    distances_per_point = []
    for point in nominal_col_data:
        distances_to_prototype = []
        for proto in nominal_col_prototypes:
            # if point is equal to prototype, sigma function should return 0, else return 1
            dist = len(point) - sum(point == proto)
            distances_to_prototype.append(dist)
        distances_per_point.append(distances_to_prototype)

    return distances_per_point


def data_distances_to_prototype(data, prototypes, gamma):
    indeces_numeric_columns = [idx for idx, column in enumerate(data.transpose()) if is_numeric(column)]
    indeces_nominal_columns = missing_elements(indeces_numeric_columns, len(data[0]))

    if not indeces_numeric_columns:
        return gamma * np.asarray(get_distance_nominal(data, indeces_nominal_columns, prototypes))
    if not indeces_nominal_columns:
        return get_distance_numeric(data, indeces_numeric_columns, prototypes)

    numeric_distances = get_distance_numeric(data, indeces_numeric_columns, prototypes)
    nominal_distances = get_distance_nominal(data, indeces_nominal_columns, prototypes)
    dist = np.add(numeric_distances, gamma * np.asarray(nominal_distances))

    return dist


def kprototypes(x, k, iterations, gamma):
    # Randomly choosing initial prototypes
    random_indeces = np.random.choice(len(x), k, replace=False)
    prototypes = x[random_indeces, :]

    # Finding the distance between prototypes and all the data points
    # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
    distances = data_distances_to_prototype(x, prototypes, gamma)

    # Select centroid with the minimum Distance
    # Each index in the array represents each point in the dataset, and the item stored in it is the cluster
    # the point belongs to
    points_closest_prototype = np.argmin(distances, axis=1)

    # Repeat the above steps for a defined number of iterations
    for _ in range(iterations):
        new_prototypes = [mode(x[np.where(points_closest_prototype == idx)], axis=0)[0][0] for idx in range(k)]
        distances = data_distances_to_prototype(x, np.asarray(new_prototypes), gamma)

        # Prototype with the minimum Distance
        points_closest_prototype = np.argmin(distances, axis=1)

    return points_closest_prototype


def silhouette_plot(traindata, testdata):
    total_runs = []
    K = range(2, 11)
    for _ in range(5):
        performance = []
        for k in K:
            data = traindata.copy()
            prediction = kprototypes(data, k, 50, 1)
            performance.append(round(metrics.silhouette_score(testdata, prediction), 4))
        total_runs.append(performance)
    avg = np.average(total_runs, axis=0)
    plt.bar(K, avg)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


def kprototypes_algorithm(traindata, testdata, dataset, task):
    if task == 'tune':
        silhouette_plot(traindata, testdata)
    elif task == 'results':
        k, gamma = get_optimal_parameters(dataset)
        prediction = kprototypes(traindata, k, 50, gamma)

        # Calculate and print silhouette score
        val = round(metrics.silhouette_score(testdata, prediction), 4)
        print('Silhouette coefficient: ' + str(val))

        # Visualize the results
        two_dim_data = preprocessing.principal_component_analysis(testdata)
        plt.scatter(two_dim_data.iloc[:, 0], two_dim_data.iloc[:, 1], c=prediction, s=1)
        plt.show()
    else:
        print('Enter a valid task')

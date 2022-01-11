import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
from scipy.spatial.distance import cdist

import preprocessing


def get_optimal_parameters(dataset):
    if dataset == 'kropt':
        return 9
    if dataset == 'satimage':
        return 3
    if dataset == 'hypothyroid':
        return 2


def kmeans(x, k, iterations):
    # Randomly choosing initial Centroids
    random_indeces = np.random.choice(len(x), k, replace=False)
    centroids = x[random_indeces, :]

    # Finding the distance between centroids and all the data points
    # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
    distances = euclidean_distances(x, centroids)

    # Centroid with the minimum Distance
    points_closest_centroid = np.argmin(distances, axis=1)

    # Repeat the above steps for a defined number of iterations
    for _ in range(iterations):
        # Updating Centroids by taking mean of Cluster it belongs to
        data = x.copy()
        new_centroids = [np.mean(data[np.where(points_closest_centroid == idx)[0]], axis=0) for idx in range(k)]
        new_distances = euclidean_distances(x, new_centroids)

        # Centroid with the minimum Distance
        points_closest_centroid = np.argmin(new_distances, axis=1)

    return points_closest_centroid


def silhouette_plot(traindata):
    total_runs = []
    K = range(2, 11)
    for _ in range(5):
        performance = []
        for k in K:
            data = traindata.copy()
            prediction = kmeans(data, k, 50)
            performance.append(round(metrics.silhouette_score(traindata, prediction), 4))
        total_runs.append(performance)
    avg = np.mean(np.asarray(total_runs), axis=0)
    plt.bar(K, avg)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()


def kmeans_algorithm(traindata, dataset, task):
    if task == 'tune':
        silhouette_plot(traindata)
    elif task == 'results':
        k = get_optimal_parameters(dataset)
        prediction = kmeans(traindata, k, 50)

        # Calculate and print silhouette score
        val = round(metrics.silhouette_score(traindata, prediction), 4)
        print('Silhouette coefficient: ' + str(val))

        # Visualize the results
        two_dim_data = preprocessing.principal_component_analysis(traindata)
        plt.scatter(two_dim_data.iloc[:, 0], two_dim_data.iloc[:, 1], c=prediction, s=1)
        plt.show()
    else:
        print('Enter a valid task')

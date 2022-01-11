import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics
import umap
import umap.plot

def get_optimal_parameters(dataset):
    if dataset == 'kropt':
        return 10
    if dataset == 'satimage':
        return 3
    if dataset == 'hypothyroid':
        return 2


def kmeans_algorithm(x, k, iterations):
    np.random.seed(10)
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

def kmeans(traindata, plotdata, dataset, algorithm):
    k = get_optimal_parameters(dataset)
    prediction = kmeans_algorithm(traindata, k, 50)

    # Calculate and print silhouette score
    val = round(metrics.silhouette_score(traindata, prediction), 4)
    print('Silhouette coefficient: ' + str(val))

    # Visualize the results
    if algorithm == 'umap':
        mapper = umap.UMAP().fit(traindata)
        p = umap.plot.points(mapper, background='black', labels=prediction)
        umap.plot.show(p)
    else:
        plt.scatter(plotdata[:, 0], plotdata[:, 1], c=prediction, s=1)
        plt.show()

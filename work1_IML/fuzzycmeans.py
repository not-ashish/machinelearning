import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics

import preprocessing

M = 2

def get_optimal_parameters(dataset):
    if dataset == 'kropt':
        return 17
    if dataset == 'satimage':
        return 7
    if dataset == 'hypothyroid':
        return 2

def find_centroids(memberships, x):

    centroids = [np.sum(x * np.power(memb, M)[:, None], axis=0) /
                 sum(pow(memb, M)) for memb in memberships.transpose()]

    return np.asarray(centroids)

def update_memberships(distances):
    memberships = []
    for dist in distances:
        memb = []
        for num in dist:
            if num!=0:
                j=0
                sumvals = []
                for den in dist:
                    if den!=0:
                        val = pow((num / den), 2/(M-1))
                        sumvals.append(val)
                    else:
                        memb.append(0)
                        j=1
                        break
                if j==0:
                    memb.append(pow(sum(sumvals),-1))
            else:
                memb.append(1)
        memberships.append(np.asarray(memb))

    return np.asarray(memberships)


def fuzzycmeans(x, c, iterations):
    # Randomly initialize centroids
    random_indeces = np.random.choice(len(x), c, replace=False)
    centroids = x[random_indeces, :]


    # Find out the distance of each point from centroid
    # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
    distances = euclidean_distances(x, centroids)

    # Updating membership values
    memberships = update_memberships(distances)



    # Repeat the above steps for a defined number of iterations
    for _ in range(iterations):
        # Find out the centroids
        centroids = find_centroids(memberships, x, c)

        # Find out the distance of each point from centroid
        # Array of arrays: each array represents a point and inside it the distances to every centroid are stored
        distances = euclidean_distances(x, centroids)

        # Centroid with the minimum Distance
        memberships = update_memberships(distances)

    prediction = np.argmax(memberships, axis=1)
    return prediction, memberships, centroids

def performance_index(traindata):
    total_runs = []
    C = range(2, 15)
    for i in range(5):
        performance = []
        for c in C:
            data = traindata.copy()
            prediction, memberships, centroids = fuzzycmeans(data, c, 10)
            distances=euclidean_distances(traindata, centroids)

            mean=[np.mean(traindata, axis=0)]
            distmean=euclidean_distances(centroids,mean)

            sum=0
            for i in range(len(centroids)):
                for k in range(len(prediction)):
                    sum = sum + (pow(memberships[k,i], M)*(pow(distances[k,i], 2) - pow(distmean[i], 2)))
            performance.append(sum)


        total_runs.append(performance)
    avg = np.average(np.asarray(total_runs), axis=0)
    plt.plot(C, avg, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Performance Index')
    plt.show()

def fuzzycmeans_algorithm(traindata, dataset, task):
    if task == 'tune':
        performance_index(traindata)
    elif task == 'results':
        c = get_optimal_parameters(dataset)
        prediction, memberships, centroids = fuzzycmeans(traindata, c, 6)

        # Calculate and print silhouette score
        val = round(metrics.silhouette_score(traindata, prediction), 4)
        print('Silhouette coefficient: ' + str(val))

        np.concatenate((traindata,centroids))

        # Visualize the results
        two_dim_data = preprocessing.principal_component_analysis(traindata)
        plt.scatter(two_dim_data.iloc[:, 0], two_dim_data.iloc[:, 1], c=prediction, s=5)

        plt.show()
    else:
        print('Enter a valid task')

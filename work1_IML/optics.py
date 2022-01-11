import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn import metrics
import numpy as np

import preprocessing

def get_optimal_parameters(dataset):
    if dataset == 'satimage':
        return 5, 0.01, 0.05, 'manhattan', 'brute'
    if dataset == 'hypothyroid':
        return 10, 0.05, 0.05, 'manhattan', 'brute'
    if dataset == 'kropt':
        return 5, 0.01, 0.05, 'manhattan', 'brute'

def find_optimal_values(traindata, s, x, c):
    twodimdata = preprocessing.principal_component_analysis(traindata)
    metrics_list = ['chebyshev', 'euclidean', 'manhattan']
    algorithms = ['brute', 'kd_tree']

    # Defining the framework of the visualization
    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = 2

    pos = 1
    for m in metrics_list:
        for a in algorithms:
            # Building the OPTICS Clustering model
            optics_model = OPTICS(min_samples=s, xi=x, min_cluster_size=c, metric=m, algorithm=a)

            # Training the model
            prediction = optics_model.fit_predict(traindata)
            if len(np.unique(prediction)) == 1:
                print('Only one cluster is generated')
            else:
                # Calculate and print silhouette score
                val = round(metrics.silhouette_score(traindata, prediction), 4)
                print('Silhouette coefficient: ' + str(val))

            # Plotting the OPTICS Clustering
            fig.add_subplot(rows, columns, pos)

            plt.scatter(twodimdata.iloc[:, 0], twodimdata.iloc[:, 1], c=prediction, s=1)
            plt.title(m + " " + a)
            pos += 1

    plt.tight_layout()
    plt.show()

def optics(traindata, s, x, c, m, a, dataset):
    optics_model = OPTICS(min_samples=s, xi=x, min_cluster_size=c, metric=m, algorithm=a)

    # Training the model
    prediction = optics_model.fit_predict(traindata)

    if dataset != 'kropt':
        # Calculate and print silhouette score
        val = round(metrics.silhouette_score(traindata, prediction), 4)
        print('Silhouette coefficient: ' + str(val))

    # Visualize the results
    two_dim_data = preprocessing.principal_component_analysis(traindata)
    plt.scatter(two_dim_data.iloc[:, 0], two_dim_data.iloc[:, 1], c=prediction, s=1)
    plt.show()


def optics_algorithm(traindata, dataset, task):
    if task == 'tune':
        s, x, c, m, a = get_optimal_parameters(dataset)
        find_optimal_values(traindata, s, x, c)
    elif task == 'results':
        s, x, c, m, a = get_optimal_parameters(dataset)
        optics(traindata, s, x, c, m, a, dataset)
    else:
        print('Enter a valid task')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

import umap
import umap.plot
import kmeans

N_COMPONENTS = 2


def plot_covariance_matrix(matrix):
    im = plt.imshow(matrix, cmap="copper_r")
    plt.colorbar(im)
    plt.show()


def pca_algorithm_scratch(traindata, task):
    # Center the data
    traindata -= np.mean(traindata, axis=0)

    # Calculate the covariance matrix
    cov_mat = np.cov(traindata, rowvar=False)
    # plot_covariance_matrix(cov_mat)
    print('covariance matrix:')
    print(cov_mat)
    print("_" * 30)

    # Determine eigenvectors and eigenvalues
    evals, evecs = np.linalg.eigh(cov_mat)
    print('eigenvalues:')
    print(evals)

    print("_" * 30)
    print('eigenvectors:')
    print(evecs)

    # Order the values in descending order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    print("_" * 30)
    print('sorted eigenvalues:')
    print(evals)

    print("_" * 30)
    print('sorted eigenvectors:')
    print(evecs)

    # Choose the biggest eivenvalue
    eigenvector_subset = evecs[:, 0:N_COMPONENTS]

    # Project the data on the line that extends the chosen eigenvector
    reduced_data = np.dot(eigenvector_subset.T, traindata.T).T

    # Data reconstruction (PCAreconstruction = PCscores⋅Eigenvectors⊤+Mean)
    # https://stats.stackexchange.com/questions/344496/pca-reconstruction-from-a-clean-set-of-eigenvectors
    reconstructed_data = np.dot(reduced_data, eigenvector_subset.T)
    reconstructed_data += np.mean(traindata, axis=0)
    print('reconstructed data:', reconstructed_data)

    # Creating a Pandas DataFrame of reduced Dataset
    principal_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    if task == 'visualize':
        plt.scatter(principal_df['PC1'], principal_df['PC2'], s=1)
        plt.show()

    return principal_df


def pca_algorithm_builtin(traindata, task):
    pca = PCA(n_components=N_COMPONENTS)
    principalComponents = pca.fit_transform(traindata)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    print("----------")
    print(pca.components_[0, :])
    print(pca.explained_variance_)

    if task == 'visualize':
        plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], s=1)
        plt.show()

    return principalDf


def incremental_pca_algorithm_builtin(traindata, task):
    transformer = IncrementalPCA(n_components=N_COMPONENTS)
    data_transformed = transformer.fit_transform(traindata)
    principalDf = pd.DataFrame(data=data_transformed
                               , columns=['principal component 1', 'principal component 2'])

    if task == 'visualize':
        plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], s=1)
        plt.show()

    return principalDf


def pca(data, algorithm, dataset, task):
    if algorithm == 'pcascratch':
        two_dim_data = pca_algorithm_scratch(data, task)
        if task == 'cluster':
            # with original data
            kmeans.kmeans(data, two_dim_data.to_numpy(), dataset, algorithm)

            # after reducing dimensionality of data
            kmeans.kmeans(two_dim_data.to_numpy(), two_dim_data.to_numpy(), dataset, algorithm)

    elif algorithm == 'ipcabuiltin':
        two_dim_data = incremental_pca_algorithm_builtin(data, task)
        if task == 'cluster':
            # with original data
            kmeans.kmeans(data, two_dim_data.to_numpy(), dataset, algorithm)

            # after reducing dimensionality of data
            kmeans.kmeans(two_dim_data.to_numpy(), two_dim_data.to_numpy(), dataset, algorithm)


    elif algorithm == 'pcabuiltin':
        two_dim_data = pca_algorithm_builtin(data, task)
        if task == 'cluster':
            # with original data
            kmeans.kmeans(data, two_dim_data.to_numpy(), dataset, algorithm)

            # after reducing dimensionality of data
            kmeans.kmeans(two_dim_data.to_numpy(), two_dim_data.to_numpy(), dataset, algorithm)

    elif algorithm == 'umap':
        if task == 'visualize':
            mapper = umap.UMAP().fit(data)
            p = umap.plot.points(mapper, background='black')
            umap.plot.show(p)
        elif task == 'cluster':
            # with original data
            kmeans.kmeans(data, dataset, algorithm)

            # after reducing dimensionality of data
            two_dim_data = pca_algorithm_builtin(data, task)
            kmeans.kmeans(two_dim_data.to_numpy(), dataset, algorithm)

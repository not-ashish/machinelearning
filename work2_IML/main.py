import sys
from scipy.io import arff

import pca
import preprocessing


def read_command_line():
    algorithm = sys.argv[1]  # pcascratch or ipcabuiltin or pcabuiltin or umap
    dataset = sys.argv[2] # satimage or kropt or hypothyroid
    task = sys.argv[3]  # visualize or cluster
    return algorithm, dataset, task


def main():
    algorithm, dataset, task = read_command_line()
    data, meta = arff.loadarff("./datasets/" + dataset + ".arff")
    cleaned_data = preprocessing.preprocessing(data, meta.types(), True)
    pca.pca(cleaned_data.to_numpy(dtype='float64'), algorithm, dataset, task)


if __name__ == "__main__":
    main()

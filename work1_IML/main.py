import sys
from scipy.io import arff

import kmeans
import optics
import preprocessing
import kprototypes
import fuzzycmeans


def read_command_line():
    algorithm = sys.argv[1]
    dataset = sys.argv[2]
    task = sys.argv[3]
    return algorithm, dataset, task


def main():
    algorithm, dataset, task = read_command_line()
    data, meta = arff.loadarff("./datasets/" + dataset + ".arff")

    if algorithm == 'optics':
        cleaned_data = preprocessing.preprocessing(data, meta.types(), True)
        optics.optics_algorithm(cleaned_data, dataset, task)
    elif algorithm == 'kmeans':
        cleaned_data = preprocessing.preprocessing(data, meta.types(), True)
        kmeans.kmeans_algorithm(cleaned_data.to_numpy(), dataset, task)
    elif algorithm == 'kprototypes':
        cleaned_data = preprocessing.preprocessing(data, meta.types(), False)
        test_cleaned_data = preprocessing.preprocessing(data, meta.types(), True)
        kprototypes.kprototypes_algorithm(cleaned_data.to_numpy(), test_cleaned_data.to_numpy(), dataset, task)
    elif algorithm == 'fuzzycmeans':
        cleaned_data = preprocessing.preprocessing(data, meta.types(), True)
        fuzzycmeans.fuzzycmeans_algorithm(cleaned_data.to_numpy(), dataset, task)
    else:
        print('Enter a valid algorithm and dataset name')


if __name__ == "__main__":
    main()

import sys
from scipy.io import arff
import os
import pandas as pd

import preprocessing
import ibl


def read_command_line():
    algorithm = sys.argv[1]  # ib1 or ib2 or ib3 or kibl
    dataset = sys.argv[2]  # satimage or hypothyroid
    return algorithm, dataset


def split_data(indeces_train, indeces_test, data_df):
    datasets_train_clean = []
    start = 0
    for end in indeces_train:
        end += start
        dataset_train = data_df[start:end]
        dataset_train = dataset_train.reset_index(drop=True)
        datasets_train_clean.append(dataset_train)
        start = end

    datasets_test_clean = []
    for end in indeces_test:
        end += start
        dataset_test = data_df[start:end]
        dataset_test = dataset_test.reset_index(drop=True)
        datasets_test_clean.append(dataset_test)
        start = end

    return datasets_train_clean, datasets_test_clean


def get_datasets(dataset, algorithm):
    directory = "./datasets/" + dataset + "/"
    datasets_train = []
    datasets_test = []
    indeces_train = []
    indeces_test = []
    for filename in os.listdir(directory):
        data, meta = arff.loadarff(directory + filename)
        if 'train' in filename:
            datasets_train.append(pd.DataFrame(data))
            indeces_train.append(data.shape[0])
        elif 'test' in filename:
            datasets_test.append(pd.DataFrame(data))
            indeces_test.append(data.shape[0])

    all_data = pd.concat([pd.concat(datasets_train, ignore_index=True), pd.concat(datasets_test, ignore_index=True)],
                         ignore_index=True)
    if algorithm == 'kibl' and dataset == 'hypothyroid':
        cleaned_all_data = preprocessing.preprocessing(all_data, meta.types(), False)
    else:
        cleaned_all_data = preprocessing.preprocessing(all_data, meta.types(), True)
    datasets_train_clean, datasets_test_clean = split_data(indeces_train, indeces_test, cleaned_all_data)

    return datasets_train_clean, datasets_test_clean


def main():
    algorithm, dataset = read_command_line()
    datasets_train, datasets_test = get_datasets(dataset, algorithm)
    ibl.instance_based_algorithms(datasets_train, datasets_test, algorithm)


if __name__ == "__main__":
    main()

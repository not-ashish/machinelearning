import numpy as np
from pandas.api.types import is_numeric_dtype
import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def canberra_distances(u, v):
    with np.errstate(invalid='ignore'):
        distances = []
        for row1 in u:
            dist = []
            for row2 in v:
                dist.append(np.sum(abs(row1 - row2) / abs(row1 + row2)))
            distances.append(np.nan_to_num(np.array(dist)))

        return np.array(distances)


def vdm(index_col, a, b, train, test):

    ax = train[train[:,index_col] == a]
    ay = test[test[:,index_col] == b]
    nx = len(ax)
    ny = len(ay)
    sum = 0
    for clase in set(train[:,-1]):
        nxc = (ax[:,-1] == clase).sum()
        nyc = (ay[:,-1] == clase).sum()

        sum += math.pow(((nxc / nx) - (nyc / ny)), 2)

    return math.sqrt(sum)


def get_distance(row1, row2, types, std_numerical_columns, train, test):
    index = 0
    distance = 0
    idx = 0
    for a, b, type in zip(row1, row2, types):
        if type:  # Numeric
            try:
                distance += (abs(a - b) / (4 * std_numerical_columns[index])) ** 2
            except:
                distance += 0
            index += 1
        else:  # Nominal
            distance += vdm(idx, a, b, train, test) ** 2
        idx +=1

    return math.sqrt(distance)


def hvdm(train, test):
    types = [is_numeric_dtype(train[col]) for col in train.iloc[:, :-1]]
    std_numerical_columns = list(train.std())
    distances = []
    train = np.array(train)
    test = np.array(test)
    for row1 in train[:, :-1]:
        dist = []
        for row2 in test[:, :-1]:
            dist.append(get_distance(row1, row2, types, std_numerical_columns, train, test))
        distances.append(dist)
    return distances

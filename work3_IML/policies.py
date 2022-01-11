import numpy as np


def most_voted_solution(test, nearest_neighbor_labels):
    predictions = []
    for i in range(len(test)):
        unique, index, counts = np.unique(nearest_neighbor_labels[:, i], return_counts=True, return_index=True)
        index = np.argsort(index)
        unique = unique[index]
        counts = counts[index]
        predictions.append(unique[np.argmax(counts)])
    return predictions


def modified_plurality(test, nearest_neighbor_labels):
    predictions = []
    for i in range(len(test)):
        unique, index, counts = np.unique(nearest_neighbor_labels[:, i], return_counts=True, return_index=True)
        winner = np.argwhere(counts == np.amax(counts))
        while len(winner) > 1:
            nearest_neighbor_labels = nearest_neighbor_labels[:-1, :]
            unique, index, counts = np.unique(nearest_neighbor_labels[:, i], return_counts=True,
                                              return_index=True)
            winner = np.argwhere(counts == np.amax(counts))
        predictions.append(unique[winner[0]][0])
    return predictions


def borda(test, nearest_neighbor_labels, k):
    predictions = []
    for i in range(len(test)):
        votes = []
        for j in range(len(nearest_neighbor_labels)):
            for _ in range(k - j, 0, -1):
                votes.append(nearest_neighbor_labels[j][i])
        unique, index, counts = np.unique(votes, return_counts=True, return_index=True)
        index = np.argsort(index)
        unique = unique[index]
        counts = counts[index]
        predictions.append(unique[np.argmax(counts)])
    return predictions

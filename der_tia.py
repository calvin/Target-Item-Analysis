from __future__ import division
from itertools import combinations

from numpy import mean
from numpy import array
from scipy.stats.stats import pearsonr as pearson_correlation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def degsim(matrix, k):
    """The average Pearson correlation of the `k` nearest neightbours.
    `k` is the number of neightbours.
    """
    assert k > 1
    neigh = NearestNeighbors(k)

    neigh.fit(matrix)
    kneighbors = neigh.kneighbors(return_distance=False)

    def compute(hood):
        return mean([pearson_correlation(matrix[x], matrix[y])[0]
            for x, y in combinations(hood, r=2)])

    return [compute(hood) for hood in kneighbors]


def rdma(matrix):
    """RDMA measures the deviation of agreement from other data on a set of
    target items, combining with the inverse rating frequency for these items.
    """
    def compute(row):
        return mean([abs(value - mean(matrix[:,i])) / len(matrix[:,i])
            for i, value in enumerate(row)])

    return array([[compute(row)] for row in matrix])
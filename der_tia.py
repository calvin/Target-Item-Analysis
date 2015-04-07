from __future__ import division
from itertools import combinations, izip

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

    return array([[compute(hood)] for hood in kneighbors])


def rdma(matrix):
    """RDMA measures the deviation of agreement from other data on a set of
    target items, combining with the inverse rating frequency for these items.
    """
    def compute(row):
        return mean([abs(value - mean(matrix[:,i])) / len(matrix[:,i])
            for i, value in enumerate(row)])

    return array([[compute(row)] for row in matrix])


def find_suspicious_data(matrix):
    """Phase 1:
    extract attributes using RMDA & DegSim' use k-menas (with `k`=2) to split
    each group of attributes; choose the two greater parts, and we consider
    the intersection between the two parts.
    """
    rdma_kmeans = KMeans(2)
    rdma_rv = rdma(matrix)
    rdma_kmeans.fit(rdma_rv)
    count_0 = (rdma_kmeans.labels_ == 0).sum()
    count_1 = len(rdma_kmeans.labels_) - count_0
    suspected_cluster = 0 if count_0 > count_1 else 1

    rdma_suspects = [l == suspected_cluster for l in rdma_kmeans.labels_]

    degsim_kmeans = KMeans(2)
    degsim_rv = degsim(matrix, 10)
    degsim_kmeans.fit(degsim_rv)
    count_0 = (degsim_kmeans.labels_ == 0).sum()
    count_1 = len(degsim_kmeans.labels_) - count_0
    suspected_cluster = 0 if count_0 > count_1 else 1

    degsim_suspects = [l == suspected_cluster for l in degsim_kmeans.labels_]

    return [x and y for x, y in izip(rdma_suspects, degsim_suspects)]
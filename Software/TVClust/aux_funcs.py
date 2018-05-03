import numpy as np
from numpy import matlib
import networkx as nx

def CalRandIdx(y, SM0):
    np.fill_diagonal(SM0, 0)
    n = len(y)
    outputSM = (np.repeat(y, n) == np.matlib.repmat(y, 1, n)[0]) * 1
    np.reshape(outputSM, (n,n))
    return (np.sum(SM0 == outputSM)) / n / (n - 1)


def entropy(freqs):
    freqs = np.asarray(freqs) / float(np.sum(freqs))
    if all(freqs > 0.0):
        H = -np.sum(freqs * np.log(freqs))
    else:
        H = 0

    H = H / np.log(2)
    return H

def GetGeodesicDist(DistEuclid, epsilon):
    DistEuclid[DistEuclid > epsilon] = 99999 ########################333
    g = nx.from_numpy_matrix(DistEuclid)
    GeoDist = nx.floyd_warshall_numpy(g)
    return GeoDist

def CalRandIdx2(x, y):
    n = len(x)
    SMx = (np.repeat(x, n) == np.matlib.repmat(x, 1, n)[0]) * 1
    np.reshape(SMx, (n,n))
    SMy = (np.repeat(y, n) == np.matlib.repmat(y, 1, n)[0]) * 1
    np.reshape(SMy, (n, n))
    return (np.sum(SMx == SMy) - n) / n / (n - 1)

def fillSM(SM):

    n = np.shape(SM)[0]
    for i in range(n - 1):
        posOne = np.where(SM[i, (i + 1): ] == 1)[0] + i
        nOne = len(posOne)
        tmp1 = np.repeat(posOne, nOne).T
        tmp2 = np.matlib.repmat(posOne, 1, nOne)[0].T
        tmpCombine = np.concatenate((tmp1, tmp2), 1)
        rows = (tmpCombine[:, 1] < tmpCombine[:, 2] ).T

        if np.sum(rows) > 0:
            index = tmpCombine[rows, ]
            if type(index) == "array":
                index = np.matrix( index, (1, 2))
            K = np.shape(index)[0]
            for k in range(K):
                SM[index[k, 1], index[k, 2]] = 1

    for i in range(n):
        for j in range(i+1, n):
            SM[j, i] = SM[i, j]

    np.fill_diagonal(SM, 0)
    return SM
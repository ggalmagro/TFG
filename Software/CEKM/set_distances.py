import numpy as np
from numpy import matlib


def set_distances(x, F, g):
    n, nb_att = np.shape(x)
    nb_foc, K = np.shape(F)

    gplus = np.zeros((nb_foc - 1, nb_att))

    for i in range(1, nb_foc):
        fi = np.array([F[i, :]])
        truc = np.matlib.repmat(fi.conj().T, 1, nb_att)
        gplus[i-1,:] = np.sum(g * truc, axis = 0) / np.sum(truc, axis = 0)

    Splot = []
    S = []

    for i in range(K):
        Splot.append(np.identity(nb_att))
        S.append(np.identity(nb_att))

    s_mean = []

    for i in range(nb_foc - 1):

        aux = np.zeros((nb_att, nb_att))

        for j in range(K):
            aux = aux + np.dot(F[i + 1, j], S[j])

        s_mean.append(aux / np.max(np.sum(F[i + 1, :], 0)))

    D = np.zeros((n, nb_foc-1))

    for j in range(nb_foc - 1):

        aux = (x - np.dot(np.ones((n, 1)), np.matrix(gplus[j, :])))
        B = np.diag(np.dot(aux, aux.conj().T))

        D[:, j] = B

    return D, Splot, s_mean

import numpy as np
from numpy import matlib


def set_distances(x, F, g, m, alpha, distance):
    n, nb_att = np.shape(x)
    nb_foc, K = np.shape(F)

    beta = 2

    gplus = np.zeros((nb_foc - 1, nb_att))

    for i in range(1, nb_foc):
        fi = np.array([F[i, :]])
        truc = np.matlib.repmat(fi.conj().T, 1, nb_att)
        gplus[i-1,:] = np.sum(g * truc, axis = 0) / np.sum(truc, axis = 0)

    if (distance == 0):

        Splot = []
        S = []

        for i in range(K):
            Splot.append(np.identity(nb_att))
            S.append(np.identity(nb_att))
    else:

        S = []
        Splot = []

        ind = np.where(np.sum(F, 1) == 1)[0]

        for i in range(len(ind)):

            denom_splot = 0
            ind_i = ind[i]

            Sigma_i = np.zeros((nb_att, nb_att))

            for k in range(n):

                omega_i = np.matlib.repmat(F[ind_i, :], nb_foc, 1)
                ind_Aj = np.where(np.sum(omega_i.tolist() and F.tolist(), 1) > 0)[0]

                for j in range(len(ind_Aj)):
                    ind_j = ind_Aj[j]
                    aux = x[k, :] - gplus[ind_j - 1, :]
                    Sigma_i += (np.sum(F[ind_j, :]) ** (alpha - 1)), (m[k, ind_j - 1] ** beta) * np.dot(aux.conj().T,
                                                                                                        aux)
                    denom_splot = denom_splot + (np.sum(F[ind_j, :]) ** (alpha - 1))*(m[k, ind_j - 1] ** beta)

            if np.linalg.det(Sigma_i) != 0:
                Si = np.dot(np.linalg.det(Sigma_i) ** (1 / nb_att), np.linalg.inv(Sigma_i))
            else:
                Si = np.dot(np.linalg.det(Sigma_i) ** (1 / nb_att), np.linalg.pinv(Sigma_i))

            Splot.append(Sigma_i / denom_splot)
            S.append(Si)

    s_mean = []

    for i in range(nb_foc - 1):

        aux = np.zeros((nb_att, nb_att))

        for j in range(K):
            aux = aux + np.dot(F[i + 1, j], S[j])

        s_mean.append(aux / np.max(np.sum(F[i + 1, :], 0)))

    D = np.zeros((n, nb_foc-1))

    for j in range(nb_foc - 1):

        aux = (x - np.dot(np.ones((n, 1)), np.matrix(gplus[j, :])))

        if (distance == 0):
            B = np.diag(np.dot(aux, aux.conj().T))
        else:
            var = s_mean[j]
            B = np.diag(np.dot(np.dot(aux, var), aux.conj().T))

        D[:, j] = B

    return D, Splot, s_mean

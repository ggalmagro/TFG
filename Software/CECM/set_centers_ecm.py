import numpy as np
from numpy import matlib


def set_centers_ecm(x, m, F, s_mean, alpha, beta):

    nb_foc, K = np.shape(F)
    n, nb_att = np.shape(x)

    c = np.sum(F[1:, :], 1)
    ind_singleton = np.where(c == 1)[0] + 1

    for l in range(K):

        indl = ind_singleton[l]


        for i in range(n):

            Ril = np.zeros((nb_att, nb_att))
            Fl = np.matlib.repmat(F[indl, :], nb_foc, 1)
            ind_aj = np.where(np.sum(np.logical_and(Fl, F), axis=1) == c[indl - 1])[0] -1

            for j in range(len(ind_aj)):
                Ril = Ril + c[ind_aj[j]] ** (alpha - 1) * m[i, ind_aj[j]] ** beta * s_mean[ind_aj[j]]

            if i == 0:
                Rl = Ril
            else:
                Rl = np.concatenate((Rl, Ril))

        if l == 0:
            R = Rl
        else:
            R = np.concatenate((R, Rl), axis=1)

        for k in range(K):

            Bkl = np.zeros((nb_att, nb_att))

            for i in range(n):

                indk = ind_singleton[k]
                Fl = np.matlib.repmat(np.sign(F[indl, :] + F[indk, :]), nb_foc, 1)
                ind_aj = np.where(np.sum(np.logical_and(Fl, F), axis=1) == sum(Fl[0, :]))[0] - 1

                for j in range(len(ind_aj)):
                    Bkl = Bkl + c[ind_aj[j]] ** (alpha - 2) * m[i, ind_aj[j]] ** beta * s_mean[ind_aj[j]]

            if k == 0:
                Bl = Bkl
            else:
                Bl = np.concatenate((Bl, Bkl))


        if l == 0:
            B = Bl
        else:
            B = np.concatenate((B, Bl), axis=1)

    X = (x.conj().T).reshape(n * nb_att, 1)
    g = np.linalg.solve(B.conj().T, np.dot(R.conj().T, X))

    return (g.reshape(nb_att, K)).conj().T

import numpy as np
import scipy.linalg as SCLA

def CSPK(L, Q, D_norm, vol, K):

    N = np.shape(L)[0]

    lam = np.linalg.svd(Q)[1][:2*K]
    beta = (lam[K] + lam[K-1])/2 - 10**(-6)

    Q1 = Q - beta*np.identity(N)

    vec = SCLA.eig(L, Q1)[1]

    for i in range(N):
        vec[:,i] = vec[:,i]/np.linalg.norm(vec[:,i])

    satisf = np.diag(np.dot(np.dot(vec.conj().T, Q1), vec))
    I = np.where(satisf >= 0)[0]


    cost = np.diag(np.dot(np.dot(vec[:,I].conj().T, L), vec[:,I]))

    ind = np.argsort(cost)

    not_done = True
    i = 0
    while not_done:

        if (vec[:, I[ind[i]]] > 0).sum() != 0 and (vec[:, I[ind[i]]] < 0).sum() != 0:
            not_done = False

        i += 1

    #Revisar linea
    ind = np.delete(ind, [0,i-2])

    ind = ind[:min(len(ind), K-1)]
    cost = cost[ind]
    U = vec[:, I[ind]]

    for i in range(np.shape(U)[1]):
        U[:, i] = np.dot(D_norm, (U[:, i] * vol**(1/2)) * (1 - cost[i]))

    return U
import numpy as np
from numpy import matlib
import copy

def get_bit(numb, K, ind):
    return int((''.join(str(1 & int(numb) >> i) for i in range(K)))[ind])

def CEVCLUS(D, c, link, Xi, version):

    n = np.shape(D)[0]

    W = 1 - np.identity(n)
    I = np.where(W != 0)
    D1 = D[I]

    if version == 3:
        F = np.identity(c)
    if version == 2:
        F = np.concatenate((np.identity(c), np.zeros((1,c))))
        F = np.concatenate((F, np.ones((1,c))))
    if version == 1:
        ii = np.array(range(0, 2**c))
        F = np.zeros((len(ii), c))
        for i in range(c):
            F[:, i] = [get_bit(j, c, i) for j in ii.conj().T]

    f = np.shape(F)[0]
    xi = np.zeros((f, f))

    for i in range(f):
        for j in range(f):
            xi[i, j] = 1 - np.max(np.min(F[[i, j], :], 0))

    card = np.sum(F, 1).conj().T
    i0 = np.where(card == 0)[0]
    if len(i0) != 0:
        card[i0] = c

    alpha = 0.1*np.random.rand(n, f)
    alpha = np.matrix([[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]])
    K = conflict(alpha, xi)[0]
    K1 = K[I]
    ab = np.matrix([[1], [0]])

    param = np.concatenate((np.reshape(alpha, (n*f, 1)), ab[:]))
    options = [1, 2000, 1e-6, 20]

    P, J, Jev, Jc, it = harris(param, options, D, [], n, Xi, xi, card, W)
    param = P

    P, J, Jev, Jc, it = harris(param, options, D, link, n, Xi, xi, card, W)

    alpha = np.reshape(P[:n*f], (n, f))
    K, mass = conflict(alpha, xi)
    ab = P[n*f:]
    BetP = mtobetpF_(mass, F)

    return mass, BetP, J, ab

def harris(x, options, *args):
    pas = 0.1 * np.ones(np.shape(x))
    a = 1.2
    b = 0.8
    c = 0.5
    ovf = 1e4
    unf = 1e-6
    err = []
    it = 0
    gain = float(1)

    yp, yev, yc, gp = coutmds(x, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    xp = x
    x = xp - np.multiply(pas, gp)
    g = gp
    itValid = 0
    histY = []
    histPri = []

    while gain >= options[2] and it <= options[1]:
        it += 1
        y, yev, yc, gp = coutmds(x, args[0], args[1], args[2], args[3], args[4], args[5], args[6])
        if y > yp:
            x = xp
            g = gp
            pas = pas * c
            x = x - np.multiply(pas, g)
            itValid = itValid + 1
        else:
            gain = 0.9 * gain + 0.1 * np.abs(yp-y)
            xp = x
            test = (np.multiply(g, gp)) >= 0

            pas = np.multiply((test * a) + ((~test) * b), pas)
            pas = np.multiply(pas <= ovf, pas) + (pas > ovf) * ovf
            pas = np.multiply(pas >= unf, pas) + (pas < unf) * unf
            gp = g
            x = xp - np.multiply(pas, g)
            yp = y

    return x, y, yev, yc, itValid

def coutmds(param,D,link,n,Xi,xi,card,W):

    f = np.shape(xi)[1]
    alpha = np.reshape(param[:n*f], (n,f))
    K, mass = conflict(alpha, xi)
    ab = param[n*f:]

    # if len(link) != 0:
    #     ptsLink = np.unique(link[:, :2])
    # else:
    #     ptsLink = []

    I = np.where(W != 0)
    K1 = np.matrix(K[I]).T
    D1 = np.matrix(D[I]).T
    C = 1/np.sum(D1)

    if np.min(D1) == 0:
        cst = 0.1*np.mean(D1)
    else:
        cst = 0

    D = D + 100 * np.identity(n)
    I = C * np.sum(np.power((int(ab[0]) * K1 + ab[1] - D1), 2) / (D1 + cst))

    if len(link) != 0:
        CLlink = np.matrix(link[np.where(link[:, 2] == 0)[0], :2], dtype=np.int8)
        MLlink = np.matrix(link[np.where(link[:, 2] == 1)[0], :2], dtype=np.int8)
    else:
        CLlink = []
        MLlink = []

    emptysetPos = np.where(np.sum(xi, 0) == f)[0]

    if len(link) != 0: #Check this if
        not_xi = np.array(~np.array(xi, dtype=bool), np.int8)
        mixj = np.concatenate((mass[link[:, 0], emptysetPos], mass[link[:, 1], emptysetPos]), 1)
        xak = np.zeros(np.shape(xi))
        aux = np.where(np.sum(not_xi, 0) == 2)[0]
        xak[aux, aux] = 1
        Pml = 1 - (np.sum(mixj, 1) - np.multiply(mixj[:,0], mixj[:,1])) - \
              np.matrix(np.diag(np.dot(np.dot(mass[link[:,0].flatten()[0], :], xak), mass[link[:,1].flatten()[0], :].conj().T))).T

        Pcl = np.matrix(np.diag(np.dot(np.dot(mass[link[:,0].flatten()[0], :], not_xi), mass[link[:,1].flatten()[0], :].conj().T))).T
        coefP = copy.deepcopy(link[:,2])
        coefP[coefP == 0] = -1
        P = np.multiply(coefP, Pml) + 1 - np.multiply(coefP, Pcl)
    else:
        P = 0

    Jev = I
    Jc = np.mean(P)
    J = I + Xi * np.mean(P)

    gradalpha = np.zeros((n,f))

    for k in range(n):
        ek = np.multiply(np.matrix(W[k, :]), (ab[0] * K[k, :] + ab[1] - D[k, :]))
        ek = ek/(D[k,:]+cst)
        ek[0,k] = 0
        ek = ek.conj().T
        A = np.dot(mass[k, :].conj().T, mass[k, :])
        v = np.multiply(mass[k, :], (mass[k, :] - 1))
        np.fill_diagonal(A, v)
        B = np.dot(xi, mass.conj().T)
        dsk = np.dot(B.conj().T, A)
        dsk = np.multiply(np.multiply(ab[0], dsk), np.matlib.repmat(ek, 1, f))
        G = np.sum(dsk, 0)
        PartOne = 4 * np.dot(C, G)

        gradj = 0

        if len(link) != 0:
            pos1, pos2 = np.where(link[:, :2] == k)
            if len(pos1) != 0 and len(pos2) != 0:
                uniques = np.unique(np.array(link[pos1, :2]))
                indL = uniques[uniques != k]
                miemptyset = np.matlib.repmat(A[emptysetPos, :], len(indL), 1)
                mjemptyset = np.matlib.repmat(mass[indL, emptysetPos].T, 1, f)
                not_xi = np.array(~np.array(xi, dtype=bool), np.int8)
                cache = np.matlib.repmat((np.sum(not_xi, 0) == 2), f, 1)
                gradCLj = np.dot(np.dot(mass[indL, :], (not_xi)), A)
                gradMLj = np.multiply(miemptyset, (mjemptyset - 1)) - (np.dot(np.multiply(A, cache), mass[indL, :].conj().T)).conj().T

                aux = np.concatenate((link[pos1, :2] + 1, link[pos1, 2] - 1), axis=1)
                orderLink = np.reshape(aux, (1, np.shape(aux)[0]*np.shape(aux)[1]), 'A')
                orderLink = orderLink[np.where(orderLink != k + 1)]
                orderLink = np.reshape(orderLink, (2, np.shape(orderLink)[1]/2))
                coefP = np.matlib.repmat(orderLink[:, 1] + 1, 1, f)
                coefP[coefP == 0] = -1

                gradj = np.multiply(coefP, gradMLj) - np.multiply(coefP, gradCLj)
                #gradj = gradMLj - gradCLj

        gradalpha[k, :] = PartOne + np.dot(Xi, (np.sum(gradj, 0))/(np.maximum(1, (np.shape(CLlink)[0] + np.shape(MLlink)[0])*2)))

    grada = 2 * C * np.sum(np.multiply((int(ab[0])*K1 + ab[1]- D1), K1) / (D1 + cst))
    gradb = 2 * C * np.sum((int(ab[0]) * K1 + ab[1] - D1) / (D1 + cst))
    gradJ = np.matrix(np.reshape(gradalpha, (n*f, 1), 'F'))
    gradJ = np.append(gradJ, [[grada]], 0); gradJ = np.append(gradJ, [[gradb]], 0)

    return J, Jev, Jc, gradJ


def conflict(alpha, xi):

    f = np.shape(alpha)[1]
    mass = np.matrix(np.exp(-alpha))
    mass = mass / np.matlib.repmat(np.sum(mass, 1), 1, f)
    k = np.dot(np.dot(mass, xi), mass.conj().T)

    return k, mass

def mtobetpF_(m, F):

    nbVc, nbEf = np.shape(m)
    nbEf2, nbAt = np.shape(F)
    indVide = np.where(np.sum(F,1) == 0)[0]
    out = []

    if nbEf == nbEf2:

        out = np.zeros((nbVc, nbAt))
        cardinals = np.sum(F,1).conj().T
        cardinals[np.where(cardinals == 0)] = 1
        x = m / np.matlib.repmat(cardinals, nbVc, 1)

        for i in range(nbAt):
            ind = np.where(F[:, i] == 1)[0]
            out[:, i] = np.sum(x[:, ind], 1).T

        if len(indVide) != 0:
            out = out / np.matlib.repmat(1-m[:,indVide], 1, nbAt)

    else:
        print("Error: length of the focal set mismatch ")

    return out



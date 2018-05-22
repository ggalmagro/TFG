import numpy as np
from numpy import matlib
import skfuzzy as fuzz
from set_centers_ecm import set_centers_ecm
from set_distances import set_distances
from solqp import solqp


def get_bit(numb, K, ind):
    # Primero el menos relevante
    return int((''.join(str(1 & int(numb) >> i) for i in range(K)))[ind])


def CEKM(X, K, constraints, max_iter=300, rho=100, bal=0.5, stop_thr=1e-3, init='rand', alpha=1):

    if alpha < 1:
        alpha = 1

    if rho < 0:
        rho = 100

    if bal < 0 or bal > 1:
        bal = 0.5

    rows, cols = np.shape(X)
    ident_matrix = np.identity(rows)
    beta = 2

    # constraint matrix reformulations
    mat_contraintes = np.sign(constraints + constraints.conj().T - ident_matrix)
    aux = constraints * np.sign(constraints)
    aux = np.maximum(aux, aux.conj().T)
    mat_contraintes = mat_contraintes * aux

    # construction of the focal set matrix
    nb_foc = 2 ** K
    k = np.array(range(0, 2 ** K))
    F = np.zeros((len(k), K))

    for i in range(K):
        F[:, i] = [get_bit(j, K, i) for j in k.conj().T]

    # set Aeq and beq matrix
    aeq = np.kron(ident_matrix, np.ones((1, nb_foc)))
    beq = np.ones((rows, 1))

    # centroids inicialization
    if (init == 'rand'):
        g = np.random.rand(K, cols) * np.matlib.repmat(np.max(X) - np.min(X), K, 1) + np.matlib.repmat(np.min(X), K, 1)
    else:
        g = fuzz.cluster.cmeans(X.T, K, 2, 1e-5, 100)[0]

    # centers calculus for all the subsets
    gplus = np.zeros((nb_foc-1,cols))

    for i in range(1, nb_foc):
        fi = np.array([F[i, :]])
        truc = np.matlib.repmat(fi.conj().T, 1, cols)
        gplus[i-1,:] = np.sum(g * truc, axis = 0) / np.sum(truc, axis = 0)

    # compute euclidean distance
    D = np.zeros((rows, nb_foc-1))
    for j in range(nb_foc - 1):
        aux = (X - np.dot(np.ones((rows, 1)), np.matrix(gplus[j, :])))
        B = np.diag(np.dot(aux, aux.conj().T))
        D[:, j] = B

    # compute masses
    c = np.asarray((np.sum(F[1:, :], axis=1)))
    masses = np.zeros((rows, nb_foc - 1))

    for i in range(rows):
        for j in range(nb_foc - 1):
            vect1 = D[i, :]
            vect1 = np.dot(D[i, j], np.ones((1, nb_foc - 1)) / vect1) ** (1 / (beta - 1))
            vect2 = ((c[j] ** (alpha / (beta - 1))) * np.ones((1, nb_foc - 1))) / (c ** (alpha / (beta - 1)))
            vect3 = vect1 * vect2
            div = (np.sum(vect3) + ((c[j] ** (alpha / (beta - 1))) * D[i, j] / rho) ** (1 / (beta - 1)))

            if (div == 0):
                div = 1

            masses[i, j] = 1 / div

    masses = np.concatenate((np.abs(np.ones((rows, 1)) - np.matrix(np.sum(masses, 1)).T), np.abs(masses)), 1)
    x0 = masses.conj().T.reshape(rows * nb_foc, 1)
    D, S, Smeans = set_distances(X, F, g, masses[:, 1:nb_foc], alpha)

    # Setting f matrix
    aux = mat_contraintes - np.identity(rows)
    contraintes_ml = np.maximum(aux, np.zeros((rows, rows)))
    nb_cont_par_object = np.sum(contraintes_ml, 1)
    aux_zeros = np.zeros((nb_foc, 1))
    aux_zeros[0, 0] = 1
    fvide = np.kron(nb_cont_par_object, aux_zeros)
    f = fvide

    # Setting constraints matrix
    ind = np.tril_indices(rows, -1)
    nb_ml = len(np.where(mat_contraintes[ind] == 1)[0])
    nb_cl = len(np.where(mat_contraintes[ind] == -1)[0])

    if (nb_ml == 0):
        nb_ml = 1

    if (nb_cl == 0):
        nb_cl = 1

    ml_mat = np.power(((np.sign(np.power(np.dot(F, np.ones((K, 1))) - 1, 2))) - 1), 2)
    ml_mat = np.dot(ml_mat, ml_mat.conj().T) * np.dot(F, F.conj().T)
    cl_mat = np.sign(np.dot(F, F.conj().T))
    ml_mat = ml_mat * -np.sign(bal) / (2 * nb_ml)
    cl_mat = cl_mat * np.sign(bal) / (2 * nb_cl)

    # contraints matrix with respect to the constraints give in parameters
    aux = np.tril(mat_contraintes, -1)
    contraintes_ml = np.maximum(aux, np.zeros((rows, rows)))
    contraintes_cl = np.absolute(np.minimum(aux, np.zeros((rows, rows))))

    ml_aux = np.kron(contraintes_ml, np.ones((nb_foc, nb_foc)))
    cl_aux = np.kron(contraintes_cl, np.ones((nb_foc, nb_foc)))

    contraintes_mat = np.matlib.repmat(ml_mat, rows, rows) * ml_aux + np.matlib.repmat(cl_mat, rows, rows) * cl_aux
    contraintes_mat = contraintes_mat + contraintes_mat.conj().T

    # Setting H matrix
    aux = np.dot(D, np.concatenate((np.zeros((nb_foc - 1, 1)), np.identity(nb_foc - 1)), 1))
    aux = aux + np.concatenate((np.ones((rows, 1)) * rho, np.zeros((rows, nb_foc - 1))), 1)

    vect_dist = aux.flatten()

    card = np.sum(F.conj().T, 0)
    card[0] = 1
    card = np.matlib.repmat(card ** alpha, 1, rows)

    if (bal > 0):
        H = (1 - bal) * np.diag(vect_dist * card / (rows * nb_foc)) + bal * contraintes_mat
    else:
        H = np.diag(vect_dist.T * card / (rows * nb_foc)) + contraintes_mat

    not_finished = True
    gold = g
    it_count = 0
    while not_finished and it_count < max_iter:
        mass, l, fval = solqp(H, aeq, beq, f, x0)

        x0 = mass

        # reshape m
        m = mass.reshape(nb_foc, rows)
        m = np.asmatrix(m[1:nb_foc, :]).conj().T

        # calculation of centers
        g = set_centers_ecm(X, m, F, Smeans, alpha, beta)
        D, S, Smeans = set_distances(X, F, g, masses, alpha)

        # Setting H matrix
        aux = np.dot(D, np.concatenate((np.zeros((nb_foc - 1, 1)), np.identity(nb_foc - 1)), 1))
        aux = aux + np.concatenate((np.ones((rows, 1)) * rho, np.zeros((rows, nb_foc - 1))), 1)

        vect_dist = aux.flatten()

        card = np.sum(F.conj().T, 0)
        card[0] = 1
        card = np.matlib.repmat(card ** alpha, 1, rows)

        H = (1 - bal) * np.diag(vect_dist * card / (rows * nb_foc)) + bal * contraintes_mat

        J = np.dot(np.dot(mass.conj().T, H), mass) + bal

        diff = np.abs(g - gold)
        grater_than_threshold = diff > stop_thr
        not_finished = sum(diff[grater_than_threshold]) > 0
        gold = g
        it_count += 1

    m = np.concatenate( (np.abs(np.ones((rows, 1)) - np.sum(m, 1)), np.abs(m)), 1)

    # Bet calcul
    bet_p = np.zeros((rows, K))

    cardinals = np.array([0, 1])

    for i in range(1, K):
        cardinals = np.append(cardinals, cardinals + 1)

    cardinals[0] = 1

    aux = m / np.matlib.repmat(cardinals, rows, 1)

    for i in range(1, K+1):

        ind = np.array(range(2 ** (i-1) + 1, 2 ** i +1))

        if (i < K):
            for j in range(1,K - i + 1):
                ind = np.append(ind, ind + 2 ** (i + j - 1))

        ind = ind - 1
        sum_ind = np.sum(aux[:, ind], 1).T
        bet_p[:, i-1] = sum_ind

    predicted = np.array([np.argmax(bet_p[i, :]) for i in range(np.shape(bet_p)[0])], dtype=np.uint8)

    return predicted
    #return bet_p, m, g, J

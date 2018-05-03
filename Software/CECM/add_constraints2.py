import numpy as np
import math


def add_constraints(x, y, mat_const, nb_const, noise, prop):
    n, nb_att = np.shape(x)

    # check the number of constraint to find

    nb_const_max = math.factorial(n - 1)
    nb_const_actual = np.shape(np.nonzero(mat_const - np.identity(n)))[1]

    if nb_const_max < nb_const_actual - nb_const:

        nb_const = nb_const_max - nb_const_actual

        if np.sign(nb_const) == -1:
            print("Error: number of constraints is too high")
            return -1

    i = 0

    while i < nb_const:
        ind1 = np.random.randint(0, n)
        ind2 = np.random.randint(0, n)

        while ind1 == ind2:
            ind2 = np.random.randint(0, n)

        if mat_const[ind1, ind2] == 0:

            mat_const[ind1, ind2] = (y[ind1] == y[ind2]) * 2 - 1

            if noise > np.random.rand():
                mat_const[ind1, ind2] = int((mat_const[ind1, ind2] == -1)) * 2 - 1

            i += 1

    print (i)
    print(np.count_nonzero(mat_const - np.identity(len(y))))

    result_matrix = mat_const

    if prop == 1:

        aux1 = np.sign(mat_const + mat_const.conj().T - np.identity(n))
        aux2 = mat_const * np.sign(mat_const)
        aux2 = np.maximum(aux2, aux2.conj().T)
        mat_const = aux1 * aux2

        # Must link propagation
        ml = np.matrix(mat_const > 0, dtype= np.uint8)

        for i in range(1, nb_const):
            ml = np.matrix(ml.tolist() or np.linalg.matrix_power(ml, i))

        # Cannot link propagation
        cl = np.matrix(mat_const < 0, dtype= np.uint8)
        cl = np.matrix((cl.tolist() or np.dot(cl, ml).tolist() or np.dot(ml, cl) * 1), dtype=np.uint8)
        cl = np.matrix(cl.tolist() or np.dot(cl, ml).tolist())

        result_matrix = ml - cl

    return result_matrix

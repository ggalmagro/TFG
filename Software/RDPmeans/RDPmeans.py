import numpy as np
from scipy import spatial as sdist


def objective(x, assignment, centroid, mat_const, xi, lamb):

    for d in range(np.shape(x)[0]):

        obj = sdist.distance.euclidean(x[d, :], centroid[assignment[d], :])
        pts_in_cluster = np.where(assignment == assignment[d])[0]
        friends = 0
        strangers = 0

        for p in range(len(pts_in_cluster)):

            if mat_const[pts_in_cluster[p], d] == 1:

                friends += 1

            elif mat_const[pts_in_cluster[p], d] == -1:

                strangers += 1

        obj -= xi * (friends - strangers)

    obj = obj + lamb * np.shape(centroid)[0]

    return obj

#Suponemos que escogemos una funcion cuadratica para la divergencia de Bregman, de
#esta forma la divergencia de bregamn es igual a la distancia euclidea.


def RDPM(X, lamb, constraints, max_iter = 300, xi_0 = 0.1, xi_rate = 1):

    num, dim = np.shape(X)

    nb_clusters = 1
    data_min = np.min(X, 0)
    data_max = np.max(X, 0)
    data_diff = data_max - data_min

    centroid = np.random.rand(nb_clusters, dim)

    for i in range(nb_clusters):
        centroid[i, :] = np.multiply(centroid[i, :], data_diff)
        centroid[i, :] = centroid[i, :] + data_min

    pos_diff = 1
    xi = 0

    assignment = np.zeros(num)
    iter = 0
    iter_no_change = 1

    while (pos_diff > 0 or iter_no_change < 3) and iter < max_iter:
        if pos_diff > 0:

            iter_no_change = 0

        else:

            iter_no_change += 1

        iter += 1

        #for d in np.random.permutation(num):
        for d in range(num):
            min_diff = float("inf")
            curr_assignment = -1

            for c in range(nb_clusters):
                friends = 0
                strangers = 0

                if iter > 1:
                    pts_in_cluster = np.where(assignment == c)[0]

                    for p in range(len(pts_in_cluster)):

                        if constraints[pts_in_cluster[p], d] == 1:

                            friends += 1

                        elif constraints[pts_in_cluster[p], d] == -1:

                            strangers += 1

                diff2c = sdist.distance.euclidean(X[d, :], centroid[c, :])
                #Apply penlaty to difference
                diff2c = diff2c - xi * (friends - strangers)

                if min_diff >= diff2c:

                    curr_assignment = c
                    min_diff = diff2c

            if min_diff >= lamb:

                nb_clusters += 1
                curr_assignment = nb_clusters - 1
                centroid = np.vstack((centroid, X[d, :]))

            assignment[d] = curr_assignment

        old_positions = centroid

        centroid = np.zeros((nb_clusters, dim))
        points_in_cluster = np.zeros(nb_clusters)

        for d in range(len(assignment)):

            centroid[assignment[d], :] = centroid[assignment[d], :] + X[d, :]
            points_in_cluster[assignment[d]] += 1

        add = 0
        e = nb_clusters

        for c in range(e):

            if points_in_cluster[c] != 0:

                centroid[c - add, :] = centroid[c - add, :] / points_in_cluster[c]

            else:
                centroid = np.delete(centroid, c - add, axis = 0)
                nb_clusters -= 1
                add += 1
                ind = np.where(assignment > c)
                assignment[ind] -= 1


        if np.shape(centroid) != np.shape(old_positions):

            pos_diff = 1 #float('inf')

        else:

            pos_diff = np.sum(np.power(centroid - old_positions, 2))

        if pos_diff <= 0:
            pass
            #print ("Terminated by reaching the while threshold")

        xi = xi_0 * (xi_rate ** iter)

    return assignment, nb_clusters


























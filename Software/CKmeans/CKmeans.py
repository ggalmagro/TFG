import numpy as np
from scipy import spatial as sdist


def CKmeans(x, k, const_mat, max_iter = 20000, threshold = 0):

    num, dim = np.shape(x)

    points_in_cluster = []

    data_min = np.min(x, 0)
    data_max = np.max(x, 0)
    data_diff = data_max - data_min

    centroid = np.random.rand(k, dim)

    for i in range(k):
        centroid[i, :] = centroid[i, :] * data_diff
        centroid[i, :] = centroid[i, :] + data_min

    pos_diff = float("inf")

    iter_all = 0

    while pos_diff > threshold:
        iter_all += 1

        if iter_all >= max_iter:
            print ("Max iteration number exceeded")
            break

        assignment = []

        for d in range(num):

            min_diff = float("inf")
            curr_assignment = -1

            for c in range(k):

                diff2c = sdist.distance.euclidean(x[d, :], centroid[c, :])

                if min_diff > diff2c:

                    violation = False

                    for dd in range(len(assignment)):

                        if const_mat[d, dd] < 0 or const_mat[dd, d] < 0:

                            if assignment[dd] == c:
                                violation = True

                        if const_mat[d, dd] > 0 or const_mat[dd, d] > 0:

                            if assignment[dd] != c:
                                violation = True

                    if not violation:

                        curr_assignment = c
                        min_diff = diff2c

            if curr_assignment == -1:

                print ("No feasible assignation")
                return centroid, points_in_cluster, assignment

            else:

                assignment = np.append(assignment, [curr_assignment])

        old_positions = centroid

        centroid = np.zeros((k, dim))
        points_in_cluster = np.zeros(k)

        for d in range(len(assignment)):

            centroid[assignment[int(d)], :] += x[int(d), :]
            points_in_cluster[assignment[int(d)]] += 1

        for c in range(k):

            if points_in_cluster[c] != 0:

                centroid[c, :] = centroid[c, :] / points_in_cluster[c]

            else:

                centroid[c, :] = (np.random.rand(1, dim) * data_diff) + data_min


        if np.shape(centroid) != np.shape(old_positions):

            pos_diff = 1
        else:

            pos_diff = np.sum(np.power(centroid - old_positions, 2))

        if pos_diff <= 0:

            print ("Terminated by reaching the while threshold")

    return centroid, points_in_cluster, assignment

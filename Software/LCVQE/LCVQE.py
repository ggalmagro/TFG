import numpy as np
from numpy import matlib

def LCVQE(X, k, constraints, centroids, max_iter = 100):

    numb_obj, dim = np.shape(X)

    data_min = np.min(X, 0)
    data_max = np.max(X, 0)
    data_diff = data_max - data_min

    centroid = np.random.rand(k, dim)

    for i in range(k):
        centroid[i, :] = centroid[i, :] * data_diff
        centroid[i, :] = centroid[i, :] + data_min

    centroids = centroid

    distances = np.zeros((numb_obj, k))
    iter = 1
    old_idx = []
    idx = np.zeros(numb_obj)

    MLs = np.where(constraints[:, 2] == 1)[0]
    CLs = np.where(constraints[:, 2] == -1)[0]

    while iter <= max_iter and not np.all(old_idx == idx):

        GMLV = [[] for i in range(k)]
        GCLV = [[] for i in range(k)]

        old_idx = idx

        for c in range(k):
            diff_to_centroid = X - np.matlib.repmat(centroids[c,:], numb_obj, 1)
            distances[:, c] = np.sum(np.power(diff_to_centroid, 2), 1).T

        idx = np.argmin(distances, 1)

        for c in MLs:
            s_1 = constraints[c, 0]
            s_2 = constraints[c, 1]
            c_j = idx[s_1]
            c_n = idx[s_2]

            if c_j == c_n:
                continue

            case_a = 0.5 * (distances[s_1, c_j] + distances[s_2, c_n]) + 0.25 * (distances[s_1, c_n] + distances[s_2, c_j])
            case_b = 0.5 * distances[s_1, c_j] + 0.5 * distances[s_2, c_j]
            case_c = 0.5 * distances[s_1, c_n] + 0.5 * distances[s_2, c_n]

            idx_min = np.argmin([case_a, case_b, case_c])

            if idx_min == 0:
                GMLV[c_j] = np.append(GMLV[c_j], s_2)
                GMLV[c_n] = np.append(GMLV[c_n], s_1)
                idx[s_1] = c_j
                idx[s_2] = c_n

            elif idx_min == 1:
                idx[s_1] = c_j
                idx[s_2] = c_j

            else:
                idx[s_1] = c_n
                idx[s_2] = c_n

        if np.shape(CLs) != (0, 0):
            idx_sorted_dist = np.argsort(distances, 1)

        for c in CLs:
            s_1 = constraints[c, 0]
            s_2 = constraints[c, 1]
            c_j = idx[s_1]
            c_n = idx[s_2]

            if c_j != c_n:
                continue

            r_j = s_1
            MM_j = idx_sorted_dist[s_1, 1]
            closest_object = s_2

            if distances[s_2, c_j] > distances[s_1, c_j]:
                r_j = s_2
                MM_j = idx_sorted_dist[s_2, 1]
                closest_object = s_1

            case_a = 0.5 * distances[s_1, c_j] + 0.5 * distances[s_2, c_j] + 0.5 * distances[r_j, MM_j]
            case_b = 0.5 * distances[closest_object, c_j] + 0.5 * distances[r_j, MM_j]

            idx_min = np.argmin([case_a, case_b])

            if idx_min == 0:
                GCLV[MM_j] = np.append(GCLV[MM_j], r_j)
                idx[s_1] = c_j
                idx[s_2] = c_j

            elif idx_min == 1:
                idx[closest_object] = c_j
                idx[r_j] = MM_j

            else:
                idx[s_1] = c_j
                idx[s_2] = idx_sorted_dist[s_2, 1]

        for c in range(k):
            members = np.where(idx == c)[0]
            coords_members = np.sum(X[members, :], 0)
            coords_GMLV = np.sum(X[np.array(GMLV[c], dtype=np.int), :], 0)
            coords_GCLV = np.sum(X[np.array(GCLV[c], dtype=np.int), :], 0)
            n_j = len(members) + 0.5*len(GMLV[c]) + len(GCLV[c])
            if(n_j == 0):
                n_j = 1
            centroids[c, :] = (coords_members + 0.5*coords_GMLV + coords_GCLV) / n_j

        lcvqe = np.zeros((k, 1))

        for c in range(k):
            lcvqe[c] = 0.5*np.sum(distances[np.where(idx == c)[0], c], 0)
            sum_ML = 0
            sum_CL = 0

            for item_violated in GMLV[c]:
                sum_ML += distances[item_violated, c]

            for item_violated in GCLV[c]:
                sum_CL += distances[int(item_violated), int(c)]

            lcvqe[c] += 0.5*sum_ML + 0.5*sum_CL

        lcvqe = np.sum(lcvqe)

        iter += 1

    return idx
    #return idx, centroids, iter, lcvqe








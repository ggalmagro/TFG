from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import math

colors = ['b', 'orange', 'g', 'r', 'Brown', 'm', 'y', 'k', 'Brown', 'ForestGreen']

def draw_const(data, mat_const, ax1, ax2):

    if np.shape(mat_const)[0] != np.shape(mat_const)[1]:
        print ("Constraints matrix must be squared")
        return ax1, ax2
    else:
        size = np.shape(mat_const)[0]

    aux_const = mat_const - np.identity(size)

    for i in range(size):

        for j in range(i + 1, size):

            if aux_const[i, j] == 1:

                ax1.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], linestyle = '-', color = "black")

            elif aux_const[i, j] == -1:

                ax2.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], linestyle = '--', color = "black")

    ax1.set_title("Must-Link Constraints")
    ax2.set_title("Cannot-Link Constraints")

    return ax1, ax2


def generate_data_2D(centers, sigmas, numb_data):

    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(numb_data) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(numb_data) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(numb_data) * i))

    X = np.zeros((len(xpts) - 1, 2))
    X[:, 0] = xpts[1:]
    X[:, 1] = ypts[1:]

    y = labels[1:]

    return X, y


def draw_data_2D(data, labels, numb_labels, centroid, is_result = False):

    fig0, ax0 = plt.subplots()
    fig1, (ax1, ax2) = plt.subplots(1, 2)

    if is_result:
        first_text = 'Result clustering: '
    else:
        first_text = 'Initial clustering: '

    for label in range(numb_labels):
        ax0.plot(data[:, 0][labels == label], data[:, 1][labels == label], '.', color=colors[label])
        ax1.plot(data[:, 0][labels == label], data[:, 1][labels == label], '.', color=colors[label])
        ax2.plot(data[:, 0][labels == label], data[:, 1][labels == label], '.', color=colors[label])

    for cent in range(len(centroid)):
        ax0.plot(centroid[cent][0], centroid[cent][1], color = 'black', marker = '8')

    title = first_text + str(len(labels)) + ' points x' + str(numb_labels) + ' clusters'
    ax0.set_title(title)

    return ax0, ax1, ax2

def draw_data_2DNC(data, labels, numb_labels, title):

    fig0, ax0 = plt.subplots()

    for label in range(numb_labels):
        ax0.plot(data[labels == label, 0], data[labels == label, 1], '.', color=colors[label])

    ax0.set_title(title)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    return ax0

def gen_rand_const(X, y, mat_const, nb_const, noise = 0, prop = False):
    n, nb_att = np.shape(X)

    # check the number of constraint to find
    nb_const_max = math.factorial(n - 1)
    nb_const_actual = np.shape(np.nonzero(mat_const - np.identity(n)))[1]

    if nb_const_max < nb_const_actual - nb_const:

        nb_const = nb_const_max - nb_const_actual

        if np.sign(nb_const) == -1:
            print("Error: number of constraints is too high")
            return -1

    i = 0

    while (i < nb_const):

        alredy_used = False
        ind_mat1, ind_mat2 = np.where(mat_const - np.identity(n) != 0)

        ind1 = np.random.randint(0, n)
        ind2 = np.random.randint(0, n)

        while (ind1 == ind2):
            ind2 = np.random.randint(0, n)

        where1 = np.where(ind_mat1 == ind1)[0]
        where2 = np.where(ind_mat2 == ind1)[0]
        test_ind1 = np.concatenate((where1,where2), axis = 0)

        if len(test_ind1) != 0:  # if the new indice 1 is alredy used

            test_ind2 = np.concatenate((np.where(ind_mat2[test_ind1] == ind2)[0], np.where(ind_mat1[test_ind1] == ind2)[0]))

            if len(test_ind2) != 0:  # if the new pair is alredy used

                alredy_used = True

        if (not alredy_used):

            mat_const[ind1, ind2] = (y[ind1] == y[ind2]) * 2 - 1

            if (noise > np.random.rand()):
                mat_const[ind1, ind2] = int((mat_const[ind1, ind2] == -1)) * 2 - 1
            i += 1

    result_matrix = mat_const

    if (prop == 1):

        aux1 = np.sign(mat_const + mat_const.conj().T - np.identity(n))
        aux2 = mat_const * np.sign(mat_const)
        aux2 = np.maximum(aux2, aux2.conj().T)
        mat_const = aux1 * aux2

        # Must link propagation
        ml = (mat_const > 0)

        for i in range(1, nb_const):
            ml = np.matrix(ml.tolist() or np.linalg.matrix_power(ml, i))

        # Cannot link propagation
        cl = (mat_const < 0) * 1
        cl = np.matrix((cl.tolist() or np.dot(cl, ml).tolist or np.dot(ml, cl) * 1))
        cl = np.matrix(cl.tolist() or np.dot(cl, ml).tolist())

        result_matrix = ml - cl

    return result_matrix

def twospirals(n_points, noise=.5):
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), np.hstack((np.zeros(n_points),np.ones(n_points))))
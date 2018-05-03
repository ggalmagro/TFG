from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from add_constraints import add_constraints
from sklearn import datasets
from CKmeans import CKmeans

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

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

                ax1.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], 'k-')

            elif aux_const[i, j] == -1:

                ax2.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], 'k-')

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

    X = np.zeros((len(xpts), 2))
    X[:, 0] = xpts
    X[:, 1] = ypts

    y = labels

    return X, y


def draw_data_2D(data, labels, numb_labels, is_result = False):

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

    title = first_text + str(len(labels)) + ' points x' + str(numb_labels) + ' clusters'
    ax0.set_title(title)

    return ax0, ax1, ax2



def main():

    np.random.seed(45)

    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    #X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 300)
    ax0, ax1, ax2 = draw_data_2D(X, y, 3)

    mat_const = np.identity(len(y))
    mat_const = add_constraints(X, y, mat_const, 20, 0, 1)

    for i in range(len(y)):
        for j in range(len(y)):
            if mat_const[i, j] != mat_const[j, i]:
                print("Error de propagacion")

            if mat_const[i, j] == -1 and y[i] == y[j]:
                print("Error: cl-const incorrecta")

            if mat_const[i, j] == 1 and y[i] != y[j]:
                print("Error: ml-const incorrecta")

    draw_const(X, mat_const, ax1, ax2)

    centroid, points_in_cluster, assignment = CKmeans(X, 3, mat_const)

    ax3, ax4, ax5, = draw_data_2D(X, assignment, 3, True)
    draw_const(X, mat_const, ax4, ax5)

    plt.show()

if __name__ == "__main__": main()
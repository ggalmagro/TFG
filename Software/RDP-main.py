from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from RDPmeans.RDPmeans import RDPmeans
from functions import generate_data_2D, draw_data_2D, draw_const, add_constraints
from numpy import matlib
from sklearn.metrics import adjusted_rand_score

def main():

    np.random.seed(45)

    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    #X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 100)
    #ax0, ax1, ax2 = draw_data_2D(X, y, 4, [[0, 0], [0, 0], [0, 0]])
    mat_const = np.identity(len(y))
    mat_const = add_constraints(X, y, mat_const, int(len(y) * 0.1), 0, 1)
    T = np.mean(X, 0)
    res = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(X)[0], 1)-X)**2, 1)))
    res[::-1].sort()

    #draw_const(X, mat_const, ax1, ax2)
    centroid, points_in_cluster, assignment, nb_clusters = RDPmeans(X, 2, mat_const, 1, 0.1, 20000)
    print(res)
    ax3, ax4, ax5, = draw_data_2D(X, assignment, nb_clusters, centroid, True)
    draw_const(X, mat_const, ax4, ax5)

    plt.show()

if __name__ == "__main__": main()
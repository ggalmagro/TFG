from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from CEVCLUS.CEVCLUS import CEVCLUS
from functions import generate_data_2D, draw_data_2D, draw_const, add_constraints

def main():

    np.random.seed(45)

    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 50)
    ax0, ax1, ax2 = draw_data_2D(X, y, 3, [[4, 2], [1, 7], [5, 6]])
    X = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #mat_const = np.identity(len(y))
    #mat_const = add_constraints(X, y, mat_const, 50, 0, 1)
    # mat_const = np.matrix([[1, 0, 0], [1, 1, 1], [0, 0, 1]])
    # draw_const(X, mat_const, ax1, ax2)
    #
    # constraint_number = 50
    # list_const = np.zeros((constraint_number, 3))
    # idx = 0
    # for i in range(len(y)): #len(y)
    #     for j in range(i + 1, len(y)): #len(y)
    #         if mat_const[i, j] == 1:
    #             list_const[idx, :] = [i, j, 1]
    #             idx += 1
    #         elif mat_const[i, j] == -1:
    #             list_const[idx, :] = [i, j, 0]
    #             idx += 1

    list_const = np.matrix([[1, 2, 1], [1, 0, 1], [0, 2, 0]])
    D = np.dot(X, X.conj().T)
    N = np.dot(np.diag(np.diag(D)), np.ones(np.shape(D)))
    DistObj = np.sqrt(N + N.conj().T - 2 * D)

    mass, bet_p, J, ab = CEVCLUS(DistObj, 3, np.matrix(list_const, dtype=np.int8), 1, 1)

    #idx = np.array([np.argmax(bet_p[i, :]) for i in range(np.shape(bet_p)[0])], dtype=np.uint8)

    #ax3, ax4, ax5, = draw_data_2D(X, idx, 3, [[4, 2], [1, 7], [5, 6]], True)
    #draw_const(X, mat_const, ax4, ax5)

    plt.show()


if __name__ == "__main__": main()
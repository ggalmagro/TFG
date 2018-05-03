from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from CSPK.CSPK import CSPK
from functions import generate_data_2D, draw_data_2D, draw_const, add_constraints

def main():

    np.random.seed(45)

    iris = datasets.load_iris()

    #X = iris.data[:, :2]
    #y = iris.target

    X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 100)
    ax0, ax1, ax2 = draw_data_2D(X, y, 3, [[4, 2], [1, 7], [5, 6]])

    mat_const = np.identity(len(y))
    mat_const = add_constraints(X, y, mat_const, 150, 0, 1)
    #X = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    N = np.shape(X)[0]
    A = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            A[i, j] = np.exp(-1*np.sum(np.power(X[i,:] - X[j,:], 2) / (2*np.var(X, 0, ddof=1))))
            A[j, i] = A[i, j]

    D = np.diag(np.sum(A, 0))
    vol = np.sum(np.diag(D),0)
    D_norm = np.power(D, (-1/2))
    D_norm[D_norm == float('inf')] = 0
    L = np.identity(N) - np.dot(np.dot(D_norm, A), D_norm)

    #res = CSPK(L, np.matrix([[0,0,0], [0,0,0], [0,0,0]]), D_norm, vol, 2)
    res = CSPK(L, mat_const, D_norm, vol, 4)
    result = np.zeros((N, N))
    result[res > 0] = 1
    result[res < 0] = -1
    Q_u = np.dot(result, result.conj().T)
    #predicted_y = np.array([np.argmax(res[i, :]) for i in range(np.shape(res)[0])], dtype=np.uint8)

    draw_const(X, mat_const, ax1, ax2)

    ax3, ax4, ax5, = draw_data_2D(X, result, 3, [[4, 2], [1, 7], [5, 6]], True)
    draw_const(X, mat_const, ax4, ax5)

    plt.show()


if __name__ == "__main__": main()

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from CECM.CECM import CECM
from functions import generate_data_2D, draw_data_2D, draw_const, add_constraints


def main():
    np.random.seed(45)

    iris = datasets.load_iris()

    # X = iris.data[:, :2]
    # y = iris.target

    X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 50)
    ax0, ax1, ax2 = draw_data_2D(X, y, 3, [[4, 2], [1, 7], [5, 6]])

    mat_const = np.identity(len(y))
    mat_const = add_constraints(X, y, mat_const, 100, 0, 1)

    for i in range(len(y)):
        for j in range(len(y)):
            if mat_const[i, j] != mat_const[j, i]:
                print("Error de propagacion")

            if mat_const[i, j] == -1 and y[i] == y[j]:
                print("Error: cl-const incorrecta")

            if mat_const[i, j] == 1 and y[i] != y[j]:
                print("Error: ml-const incorrecta")

    draw_const(X, mat_const, ax1, ax2)

    m, g, bet_p, j = CECM(X, 3, mat_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)

    predicted_y = np.array([np.argmax(bet_p[i, :]) for i in range(np.shape(bet_p)[0])], dtype=np.uint8)

    ax3, ax4, ax5, = draw_data_2D(X, predicted_y, 3, [[4, 2], [1, 7], [5, 6]], True)
    draw_const(X, mat_const, ax4, ax5)

    plt.show()

if __name__ == "__main__": main()
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from TVClust.TVClust import TVClust
from functions import generate_data_2D, draw_data_2D, draw_const, gen_rand_const

def main():

    np.random.seed(45)

    iris = datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target

    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    #X, y = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 100)
    #ax0, ax1, ax2 = draw_data_2D(X, y, 3, [[4, 2], [1, 7], [5, 6]])

    mat_const = np.identity(len(y))
    mat_const = gen_rand_const(X, y, mat_const, int(len(y) * 0.25), 0, 1)
    np.savetxt('const.txt', mat_const)

    #draw_const(X, mat_const, ax1, ax2)

    mat_const2 = -1 * np.ones(np.shape(mat_const))
    checked = np.zeros(np.shape(mat_const))
    mat_const2[mat_const == 1] = 1
    mat_const2[mat_const == -1] = 0
    checked[mat_const != 0] = 1

    np.savetxt('bc_checked.txt', checked)

    result = TVClust(X, 2)

    print(result)
    ax3, ax4, ax5, = draw_data_2D(X, result, 3, [[4, 2], [1, 7], [5, 6]], True)
    draw_const(X, mat_const, ax4, ax5)

    plt.show()


if __name__ == "__main__": main()
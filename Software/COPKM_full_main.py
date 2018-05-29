from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from CECM.CECM import CEKM
from CKmeans.CKmeans import CKmeans
from LCVQE.LCVQE import LCVQE
from RDPmeans.RDPmeans import RDPM
from TVClust.TVClust import TVClust
from COPKmeans.COPKmeans import COPKM
from functions import generate_data_2D, draw_data_2D, draw_data_2DNC, draw_const, gen_rand_const, twospirals
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
import gc
import pickle
from sklearn.metrics import adjusted_rand_score


def main():

    np.random.seed(43)
    random_state = 43
    const_percent = 0.3

    iris = datasets.load_iris()
    iris_set = iris.data[:, :2]
    iris_labels = iris.target

    wine = datasets.load_wine()
    wine_set = wine.data
    wine_labels = wine.target

    breast_cancer = datasets.load_breast_cancer()
    breast_cancer_set = breast_cancer.data
    breast_cancer_labels = breast_cancer.target

    glass = fetch_mldata('Glass')
    glass_set = glass.data
    glass_labels = glass.target

    digits = datasets.load_digits()
    digits_set = digits.data
    digits_labels = digits.target

    iris = []
    wine = []
    breast_cancer = []
    glass = []
    digits = []
    gc.collect()

    rand_set, rand_labels = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 50)
    spiral_set, spiral_labels = twospirals(150)
    spiral_set += 15
    moons_set, moons_labels = datasets.make_moons(300, .5, .05, random_state)
    moons_set += 1.5
    circles_set, circles_labels = datasets.make_circles(300, .5, .05, random_state, .3)
    circles_set += 1.5

    iris_const = np.identity(len(iris_labels))
    iris_const = gen_rand_const(iris_set, iris_labels, iris_const, int(len(iris_labels) * const_percent), 0, 1)

    wine_const = np.identity(len(wine_labels))
    wine_const = gen_rand_const(wine_set, wine_labels, wine_const, int(len(wine_labels) * const_percent), 0, 1)

    breast_cancer_const = np.identity(len(breast_cancer_labels))
    breast_cancer_const = gen_rand_const(breast_cancer_set, breast_cancer_labels, breast_cancer_const,
                                         int(len(breast_cancer_labels) * const_percent), 0, 1)

    glass_const = np.identity(len(glass_labels))
    glass_const = gen_rand_const(glass_set, glass_labels, glass_const, int(len(glass_labels) * const_percent), 0, 1)

    digits_const = np.identity(len(digits_labels))
    digits_const = gen_rand_const(digits_set, digits_labels, digits_const, int(len(digits_labels) * const_percent), 0,
                                  1)

    rand_const = np.identity(len(rand_labels))
    rand_const = gen_rand_const(rand_set, rand_labels, rand_const, int(len(rand_labels) * const_percent), 0, 1)

    spiral_const = np.identity(len(spiral_labels))
    spiral_const = gen_rand_const(spiral_set, spiral_labels, spiral_const, int(len(spiral_labels) * const_percent), 0,
                                  1)

    moons_const = np.identity(len(moons_labels))
    moons_const = gen_rand_const(moons_set, moons_labels, moons_const, int(len(moons_labels) * const_percent), 0, 1)

    circles_const = np.identity(len(circles_labels))
    circles_const = gen_rand_const(circles_set, circles_labels, circles_const, int(len(circles_labels) * const_percent), 0, 1)

    iris_centroid = []
    wine_centroid = []
    breast_cancer_centroid = []
    glass_centroid = []
    digits_centroid = []
    rand_centroid = []
    spiral_centroid = []
    moons_centroid = []
    circles_centroid = []

    for i in range(np.shape(iris_const)[0]):
        for j in range(np.shape(iris_const)[1]):

            if iris_const[i, j] == 1 and iris_labels[i] != iris_labels[j]:
                print("Error en las restricciones")

            if iris_const[i, j] == -1 and iris_labels[i] == iris_labels[j]:
                print("Error en las restricciones")

    ax0 = draw_data_2DNC(iris_set, iris_labels, 3, "title")
    ax1 = draw_data_2DNC(iris_set, iris_labels, 3, "title")
    c1, c2 = draw_const(iris_set, iris_const, ax0, ax1)
    plt.show()

    ax0 = draw_data_2DNC(rand_set, rand_labels, 3, "title")
    ax1 = draw_data_2DNC(rand_set, rand_labels, 3, "title")
    c1, c2 = draw_const(rand_set, rand_const, ax0, ax1)
    plt.show()


    iris_const[iris_const == -1] = 0
    wine_const[wine_const == -1] = 0
    glass_const[glass_const == -1] = 0
    breast_cancer_const[breast_cancer_const == -1] = 0
    digits_const[digits_const == -1] = 0
    rand_const[rand_const == -1] = 0
    spiral_const[spiral_const == -1] = 0
    moons_const[moons_const == -1] = 0
    circles_const[circles_const == -1] = 0


    iris_ckm_assignment, iris_centroid = COPKM(iris_set, 3, iris_const)
    wine_ckm_assignment, wine_centroid = COPKM(wine_set, 3, wine_const)
    glass_ckm_assignment, glass_centroid = COPKM(glass_set, 6, glass_const)
    breast_cancer_ckm_assignment, breast_cancer_centroid = COPKM(breast_cancer_set, 2, breast_cancer_const)
    digits_ckm_assignment, digits_centroid = COPKM(digits_set, 10, digits_const)
    rand_ckm_assignment, rand_centroid = COPKM(rand_set, 3, rand_const)
    spiral_ckm_assignment, spiral_centroid = COPKM(spiral_set, 2, spiral_const)
    moons_ckm_assignment, moons_centroid = COPKM(moons_set, 2, moons_const)
    circles_ckm_assignment, circles_centroid = COPKM(circles_set, 2, circles_const)

    iris_ckm_rand_score = adjusted_rand_score(iris_labels, iris_ckm_assignment)
    wine_ckm_rand_score = adjusted_rand_score(wine_labels, wine_ckm_assignment)
    breast_cancer_ckm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_ckm_assignment)
    glass_ckm_rand_score = adjusted_rand_score(glass_labels, glass_ckm_assignment)
    digits_ckm_rand_score = adjusted_rand_score(digits_labels, digits_ckm_assignment)
    rand_ckm_rand_score = adjusted_rand_score(rand_labels, rand_ckm_assignment)
    spiral_ckm_rand_score = adjusted_rand_score(spiral_labels, spiral_ckm_assignment)
    moons_ckm_rand_score = adjusted_rand_score(moons_labels, moons_ckm_assignment)
    circles_ckm_rand_score = adjusted_rand_score(circles_labels, circles_ckm_assignment)

    ############################### Get scores for COP-Kmeans ###############################
    print("########################## COPK-means ##########################")
    print("Iris: " + str(iris_ckm_rand_score) +
          "\nWine: " + str(wine_ckm_rand_score) +
          "\nBreast: " + str(breast_cancer_ckm_rand_score) +
          "\nGlass: " + str(glass_ckm_rand_score) +
          "\nDigits: " + str(digits_ckm_rand_score) +
          "\nRand: " + str(rand_ckm_rand_score) +
          "\nSpiral: " + str(spiral_ckm_rand_score) +
          "\nMoons: " + str(moons_ckm_rand_score) +
          "\nCircles: " + str(circles_ckm_rand_score))

    ############################### Draw Drawable Results ###############################
    alg = "COP-KM "
    iris_plot = draw_data_2DNC(iris_set, np.asarray(iris_ckm_assignment, np.float), 3, alg + "Iris Dataset Results")
    rand_plot = draw_data_2DNC(rand_set, np.asarray(rand_ckm_assignment, np.float), 3, alg + "Rand Dataset Results")
    spiral_plot = draw_data_2DNC(spiral_set, np.asarray(spiral_ckm_assignment, np.float), 2, alg + "Spirals Dataset Results")
    moons_plot = draw_data_2DNC(moons_set, np.asarray(moons_ckm_assignment, np.float), 2, alg + "Moons Dataset Results")
    circles_plot = draw_data_2DNC(circles_set, np.asarray(circles_ckm_assignment, np.float), 2, alg + "Circles Dataset Results")
    plt.show()




if __name__ == "__main__": main()
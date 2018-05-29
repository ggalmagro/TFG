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
    const_percent = 0.25

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

    rand_set, rand_labels = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 100)
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


    iris_rdpm_assignment, iris_rdpm_nbc = RDPM(iris_set, 1.7, iris_const, 20000, 0.1, 1)
    T = np.mean(wine_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(wine_set)[0], 1) - wine_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    wine_rdpm_assignment, wine_rdpm_nbc = RDPM(wine_set, lamb_arr[2], wine_const, 20000, 0.1, 1)
    T = np.mean(breast_cancer_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(breast_cancer_set)[0], 1) - breast_cancer_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    breast_cancer_rdpm_assignment, breast_cancer_rdpm_nbc = RDPM(breast_cancer_set, lamb_arr[1], breast_cancer_const, 20000, 0.1, 1)
    T = np.mean(glass_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(glass_set)[0], 1) - glass_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    glass_rdpm_assignment, glass_rdpm_nbc = RDPM(glass_set, lamb_arr[5], glass_const, 20000, 0.1, 1)
    T = np.mean(digits_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(digits_set)[0], 1) - digits_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    digits_rdpm_assignment, digits_rdpm_nbc = RDPM(digits_set, lamb_arr[9], digits_const, 20000, 0.1, 1)

    T = np.mean(rand_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(rand_set)[0], 1) - rand_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    rand_rdpm_assignment,  rand_rdpm_nbc = RDPM(rand_set, 4, rand_const, 20000, 0.1, 1)
    T = np.mean(spiral_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(spiral_set)[0], 1) - spiral_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    spiral_rdpm_assignment, spiral_rdpm_nbc = RDPM(spiral_set, lamb_arr[1], spiral_const, 20000, 0.1, 1)
    T = np.mean(moons_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(moons_set)[0], 1) - moons_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    moons_rdpm_assignment, moons_rdpm_nbc = RDPM(moons_set, 2, moons_const, 20000, 0.1, 1)
    T = np.mean(circles_set, 0)
    lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(circles_set)[0], 1) - circles_set) ** 2, 1)))
    lamb_arr[::-1].sort()
    circles_rdpm_assignment, circles_rdpm_nbc = RDPM(circles_set, 1.5, circles_const, 20000, 0.1, 1)

    iris_rdpm_rand_score = adjusted_rand_score(iris_labels, iris_rdpm_assignment)
    wine_rdpm_rand_score = adjusted_rand_score(wine_labels, wine_rdpm_assignment)
    breast_cancer_rdpm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_rdpm_assignment)
    glass_rdpm_rand_score = adjusted_rand_score(glass_labels, glass_rdpm_assignment)
    digits_rdpm_rand_score = adjusted_rand_score(digits_labels, digits_rdpm_assignment)
    rand_rdpm_rand_score = adjusted_rand_score(rand_labels, rand_rdpm_assignment)
    spiral_rdpm_rand_score = adjusted_rand_score(spiral_labels, spiral_rdpm_assignment)
    moons_rdpm_rand_score = adjusted_rand_score(moons_labels, moons_rdpm_assignment)
    circles_rdpm_rand_score = adjusted_rand_score(circles_labels, circles_rdpm_assignment)

    ############################### Get scores for RDPM ###############################
    print("########################## RDPM ##########################")
    print("Iris: " + str(iris_rdpm_rand_score) +
          "\nWine: " + str(wine_rdpm_rand_score) +
          "\nBreast: " + str(breast_cancer_rdpm_rand_score) +
          "\nGlass: " + str(glass_rdpm_rand_score) +
          "\nDigits: " + str(digits_rdpm_rand_score) +
          "\nRand: " + str(rand_rdpm_rand_score) +
          "\nSpiral: " + str(spiral_rdpm_rand_score) +
          "\nMoons: " + str(moons_rdpm_rand_score) +
          "\nCircles: " + str(circles_rdpm_rand_score))

    ############################### Draw Drawable Results ###############################
    alg = "RDPM "
    iris_plot = draw_data_2DNC(iris_set, np.asarray(iris_rdpm_assignment, np.float), iris_rdpm_nbc, alg + "Iris Dataset Results")
    rand_plot = draw_data_2DNC(rand_set, np.asarray(rand_rdpm_assignment, np.float), rand_rdpm_nbc, alg + "Rand Dataset Results")
    spiral_plot = draw_data_2DNC(spiral_set, np.asarray(spiral_rdpm_assignment, np.float), spiral_rdpm_nbc, alg + "Spirals Dataset Results")
    moons_plot = draw_data_2DNC(moons_set, np.asarray(moons_rdpm_assignment, np.float), moons_rdpm_nbc, alg + "Moons Dataset Results")
    circles_plot = draw_data_2DNC(circles_set, np.asarray(circles_rdpm_assignment, np.float), circles_rdpm_nbc, alg + "Circles Dataset Results")
    plt.show()




if __name__ == "__main__": main()
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
    const_percent = 0.05

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

    print(np.shape(glass_set))
    print(np.shape(breast_cancer_set))

    digits = datasets.load_digits()
    digits_set = digits.data
    digits_labels = digits.target

    iris = [];
    wine = [];
    breast_cancer = [];
    glass = [];
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
    circles_const = gen_rand_const(circles_set, circles_labels, circles_const, int(len(circles_labels) * const_percent),
                                   0, 1)

    np.savetxt('breast.txt', breast_cancer_set)

    iris_labels_file = open("results/trueLabels/iris_labels.pkl", "wb")
    pickle.dump(iris_labels, iris_labels_file)
    iris_labels_file.close()

    wine_labels_file = open("results/trueLabels/wine_labels.pkl", "wb")
    pickle.dump(wine_labels, wine_labels_file)
    wine_labels_file.close()

    breast_cancer_labels_file = open("results/trueLabels/breast_cancer_labels.pkl", "wb")
    pickle.dump(breast_cancer_labels, breast_cancer_labels_file)
    breast_cancer_labels_file.close()

    glass_labels_file = open("results/trueLabels/glass_labels.pkl", "wb")
    pickle.dump(glass_labels, glass_labels_file)
    glass_labels_file.close()

    digits_labels_file = open("results/trueLabels/digits_labels.pkl", "wb")
    pickle.dump(digits_labels, digits_labels_file)
    digits_labels_file.close()

    rand_labels_file = open("results/trueLabels/rand_labels.pkl", "wb")
    pickle.dump(rand_labels, rand_labels_file)
    rand_labels_file.close()

    spiral_labels_file = open("results/trueLabels/spiral_labels.pkl", "wb")
    pickle.dump(spiral_labels, spiral_labels_file)
    spiral_labels_file.close()

    moons_labels_file = open("results/trueLabels/moons_labels.pkl", "wb")
    pickle.dump(moons_labels, moons_labels_file)
    moons_labels_file.close()

    circles_labels_file = open("results/trueLabels/circles_labels.pkl", "wb")
    pickle.dump(circles_labels, circles_labels_file)
    circles_labels_file.close()

    iris_centroid = []
    wine_centroid = []
    breast_cancer_centroid = []
    glass_centroid = []
    digits_centroid = []
    rand_centroid = []
    spiral_centroid = []
    moons_centroid = []
    circles_centroid = []

    iris_ckm_assignment, iris_centroid = COPKM(iris_set, 3, iris_const)
    wine_ckm_assignment, wine_centroid = COPKM(wine_set, 3, wine_const)
    glass_ckm_assignment, glass_centroid = COPKM(glass_set, 6, glass_const)
    breast_cancer_ckm_assignment, breast_cancer_centroid = COPKM(breast_cancer_set, 2, glass_const)
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
          " Wine: " + str(wine_ckm_rand_score) +
          " Breast: " + str(breast_cancer_ckm_rand_score) +
          " Glass: " + str(glass_ckm_rand_score) +
          " Digits: " + str(digits_ckm_rand_score) +
          " Rand: " + str(rand_ckm_rand_score) +
          " Spiral: " + str(spiral_ckm_rand_score) +
          " Moons: " + str(moons_ckm_rand_score) +
          " Circles: " + str(circles_ckm_rand_score))

    ############################### Draw Drawable Results ###############################

    iris_plot = draw_data_2DNC(iris_set, iris_ckm_assignment, 3, "COP-KM Iris Results")
    plt.show()





if __name__ == "__main__": main()
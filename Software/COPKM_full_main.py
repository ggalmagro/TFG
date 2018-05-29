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
import time



def main():

    np.random.seed(113)
    random_state = 43
    const_percent = 0.1

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

    # iris_const[iris_const == -1] = 0
    # wine_const[wine_const == -1] = 0
    # glass_const[glass_const == -1] = 0
    # breast_cancer_const[breast_cancer_const == -1] = 0
    # digits_const[digits_const == -1] = 0
    # rand_const[rand_const == -1] = 0
    # spiral_const[spiral_const == -1] = 0
    # moons_const[moons_const == -1] = 0
    # circles_const[circles_const == -1] = 0


    ml_iris_const = len(np.where(iris_const - np.identity(len(iris_labels)) == 1)[0]) / (len(iris_labels)*2)
    cl_iris_const = len(np.where(iris_const - np.identity(len(iris_labels)) == -1)[0]) / (len(iris_labels)*2)

    ml_wine_const = len(np.where(wine_const - np.identity(len(wine_labels)) == 1)[0]) / (len(wine_labels) * 2)
    cl_wine_const = len(np.where(wine_const - np.identity(len(wine_labels)) == -1)[0]) / (len(wine_labels) * 2)

    ml_glass_const = len(np.where(glass_const - np.identity(len(glass_labels)) == 1)[0]) / (len(glass_labels) * 2)
    cl_glass_const = len(np.where(glass_const - np.identity(len(glass_labels)) == -1)[0]) / (len(glass_labels) * 2)

    ml_breast_cancer_const = len(np.where(breast_cancer_const - np.identity(len(breast_cancer_labels)) == 1)[0]) / (len(breast_cancer_labels) * 2)
    cl_breast_cancer_const = len(np.where(breast_cancer_const - np.identity(len(breast_cancer_labels)) == -1)[0]) / (len(breast_cancer_labels) * 2)

    ml_digits_const = len(np.where(digits_const - np.identity(len(digits_labels)) == 1)[0]) / (len(digits_labels) * 2)
    cl_digits_const = len(np.where(digits_const - np.identity(len(digits_labels)) == -1)[0]) / (len(digits_labels) * 2)

    ml_rand_const = len(np.where(rand_const - np.identity(len(rand_labels)) == 1)[0]) / (len(rand_labels) * 2)
    cl_rand_const = len(np.where(rand_const - np.identity(len(rand_labels)) == -1)[0]) / (len(rand_labels) * 2)

    ml_spiral_const = len(np.where(spiral_const - np.identity(len(spiral_labels)) == 1)[0]) / (len(spiral_labels) * 2)
    cl_spiral_const = len(np.where(spiral_const - np.identity(len(spiral_labels)) == -1)[0]) / (len(spiral_labels) * 2)

    ml_moons_const = len(np.where(moons_const - np.identity(len(moons_labels)) == 1)[0]) / (len(moons_labels) * 2)
    cl_moons_const = len(np.where(moons_const - np.identity(len(moons_labels)) == -1)[0]) / (len(moons_labels) * 2)

    ml_circles_const = len(np.where(circles_const - np.identity(len(circles_labels)) == 1)[0]) / (len(circles_labels) * 2)
    cl_circles_const = len(np.where(circles_const - np.identity(len(circles_labels)) == -1)[0]) / (len(circles_labels) * 2)

    print("########################## ML Const Percent ##########################")
    print("Iris: " + str(ml_iris_const) +
          "\nWine: " + str(ml_wine_const) +
          "\nBreast: " + str(ml_glass_const) +
          "\nGlass: " + str(ml_breast_cancer_const) +
          "\nDigits: " + str(ml_digits_const) +
          "\nRand: " + str(ml_rand_const) +
          "\nSpiral: " + str(ml_spiral_const) +
          "\nMoons: " + str(ml_moons_const) +
          "\nCircles: " + str(ml_circles_const))

    print("########################## CL Const Percent ##########################")
    print("Iris: " + str(cl_iris_const) +
          "\nWine: " + str(cl_wine_const) +
          "\nBreast: " + str(cl_glass_const) +
          "\nGlass: " + str(cl_breast_cancer_const) +
          "\nDigits: " + str(cl_digits_const) +
          "\nRand: " + str(cl_rand_const) +
          "\nSpiral: " + str(cl_spiral_const) +
          "\nMoons: " + str(cl_moons_const) +
          "\nCircles: " + str(cl_circles_const))

    iris_start = time.time()
    iris_ckm_assignment, iris_centroid = COPKM(iris_set, 3, iris_const)
    iris_end = time.time()

    wine_start = time.time()
    wine_ckm_assignment, wine_centroid = COPKM(wine_set, 3, wine_const)
    wine_end = time.time()

    glass_start = time.time()
    glass_ckm_assignment, glass_centroid = COPKM(glass_set, 6, glass_const)
    glass_end = time.time()

    breast_cancer_start = time.time()
    #breast_cancer_ckm_assignment, breast_cancer_centroid = COPKM(breast_cancer_set, 2, breast_cancer_const)
    breast_cancer_end = time.time()

    digits_start = time.time()
    digits_ckm_assignment, digits_centroid = COPKM(digits_set, 10, digits_const)
    digits_end = time.time()

    rand_start = time.time()
    rand_ckm_assignment, rand_centroid = COPKM(rand_set, 3, rand_const)
    rand_end = time.time()

    spiral_start = time.time()
    spiral_ckm_assignment, spiral_centroid = COPKM(spiral_set, 2, spiral_const)
    spiral_end = time.time()

    moons_start = time.time()
    #moons_ckm_assignment, moons_centroid = COPKM(moons_set, 2, moons_const)
    moons_end = time.time()

    circles_start = time.time()
    circles_ckm_assignment, circles_centroid = COPKM(circles_set, 2, circles_const)
    circles_end = time.time()


    iris_ckm_rand_score = adjusted_rand_score(iris_labels, iris_ckm_assignment)
    wine_ckm_rand_score = adjusted_rand_score(wine_labels, wine_ckm_assignment)
    breast_cancer_ckm_rand_score = 0 #adjusted_rand_score(breast_cancer_labels, breast_cancer_ckm_assignment)
    glass_ckm_rand_score = adjusted_rand_score(glass_labels, glass_ckm_assignment)
    digits_ckm_rand_score = adjusted_rand_score(digits_labels, digits_ckm_assignment)
    rand_ckm_rand_score = adjusted_rand_score(rand_labels, rand_ckm_assignment)
    spiral_ckm_rand_score = adjusted_rand_score(spiral_labels, spiral_ckm_assignment)
    moons_ckm_rand_score = 0 #adjusted_rand_score(moons_labels, moons_ckm_assignment)
    circles_ckm_rand_score = adjusted_rand_score(circles_labels, circles_ckm_assignment)

    iris_elapsed_time = iris_end - iris_start
    wine_elapsed_time = wine_end - wine_start
    breast_cancer_elapsed_time = breast_cancer_end - breast_cancer_start
    glass_elapsed_time = glass_end - glass_start
    digits_elapsed_time = digits_end - digits_start
    rand_elapsed_time = rand_end - rand_start
    spiral_elapsed_time = spiral_end - spiral_start
    moons_elapsed_time = moons_end - moons_start
    circles_elapsed_time = circles_end - circles_start

    ############################### Get scores for COP-Kmeans ###############################
    print("########################## COPK-means RandIndex ##########################")
    print("Iris: " + str(iris_ckm_rand_score) +
          "\nWine: " + str(wine_ckm_rand_score) +
          "\nBreast: " + str(breast_cancer_ckm_rand_score) +
          "\nGlass: " + str(glass_ckm_rand_score) +
          "\nDigits: " + str(digits_ckm_rand_score) +
          "\nRand: " + str(rand_ckm_rand_score) +
          "\nSpiral: " + str(spiral_ckm_rand_score) +
          "\nMoons: " + str(moons_ckm_rand_score) +
          "\nCircles: " + str(circles_ckm_rand_score))

    print("########################## COPK-means Time ##########################")
    print("Iris: " + str(iris_elapsed_time) +
          "\nWine: " + str(wine_elapsed_time) +
          "\nBreast: " + str(breast_cancer_elapsed_time) +
          "\nGlass: " + str(glass_elapsed_time) +
          "\nDigits: " + str(digits_elapsed_time) +
          "\nRand: " + str(rand_elapsed_time) +
          "\nSpiral: " + str(spiral_elapsed_time) +
          "\nMoons: " + str(moons_elapsed_time) +
          "\nCircles: " + str(circles_elapsed_time))

    ############################### Draw Drawable Results ###############################
    alg = "COP-KM "
    iris_plot = draw_data_2DNC(iris_set, np.asarray(iris_ckm_assignment, np.float), 3, alg + "Iris Dataset Results")
    rand_plot = draw_data_2DNC(rand_set, np.asarray(rand_ckm_assignment, np.float), 3, alg + "Rand Dataset Results")
    spiral_plot = draw_data_2DNC(spiral_set, np.asarray(spiral_ckm_assignment, np.float), 2, alg + "Spirals Dataset Results")
    #moons_plot = draw_data_2DNC(moons_set, np.asarray(moons_ckm_assignment, np.float), 2, alg + "Moons Dataset Results")
    circles_plot = draw_data_2DNC(circles_set, np.asarray(circles_ckm_assignment, np.float), 2, alg + "Circles Dataset Results")
    plt.show()




if __name__ == "__main__": main()
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from CEKM.CEKM import CEKM
from functions import generate_data_2D, draw_data_2DNC, gen_rand_const, twospirals
from sklearn.datasets import fetch_mldata
import gc
from sklearn.metrics import adjusted_rand_score
import time


def main():

    np.random.seed(43)
    random_state = 43
    const_percent = 0.6

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

    print("Calculando Iris")
    iris_start = time.time()
    iris_cekm_assignment = CEKM(iris_set, 3, iris_const)[0]
    iris_end = time.time()

    print("Calculando Wine")
    wine_start = time.time()
    wine_cekm_assignment = CEKM(wine_set, 3, wine_const)[0]
    wine_end = time.time()

    print("Calculando Glass")
    glass_start = time.time()
    #glass_cekm_assignment = CEKM(glass_set, 2, glass_const)[0]
    glass_end = time.time()

    print("Calculando Breast Cancer")
    breast_cancer_start = time.time()
    breast_cancer_cekm_assignment = CEKM(breast_cancer_set, 2, breast_cancer_const, stop_thr=0.1)[0]
    breast_cancer_end = time.time()

    print("Calculando Digits")
    digits_start = time.time()
    #digits_cekm_assignment = CEKM(digits_set, 10, digits_const)[0]
    digits_end = time.time()

    print("Calculando Rand")
    rand_start = time.time()
    rand_cekm_assignment = CEKM(rand_set, 3, rand_const)[0]
    rand_end = time.time()

    print("Calculando Spiral")
    spiral_start = time.time()
    spiral_cekm_assignment = CEKM(spiral_set, 2, spiral_const)[0]
    spiral_end = time.time()

    print("Calculando Moons")
    moons_start = time.time()
    moons_cekm_assignment = CEKM(moons_set, 2, moons_const)[0]
    moons_end = time.time()

    print("Calculando Circles")
    circles_start = time.time()
    circles_cekm_assignment = CEKM(circles_set, 2, circles_const)[0]
    circles_end = time.time()


    iris_cekm_rand_score = adjusted_rand_score(iris_labels, iris_cekm_assignment)
    wine_cekm_rand_score = adjusted_rand_score(wine_labels, wine_cekm_assignment)
    breast_cancer_cekm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_cekm_assignment)
    glass_cekm_rand_score = "NA"  # adjusted_rand_score(glass_labels, glass_cekm_assignment)
    digits_cekm_rand_score = "NA"  # adjusted_rand_score(digits_labels, digits_cekm_assignment)
    rand_cekm_rand_score = adjusted_rand_score(rand_labels, rand_cekm_assignment)
    spiral_cekm_rand_score = adjusted_rand_score(spiral_labels, spiral_cekm_assignment)
    moons_cekm_rand_score = adjusted_rand_score(moons_labels, moons_cekm_assignment)
    circles_cekm_rand_score = adjusted_rand_score(circles_labels, circles_cekm_assignment)

    iris_elapsed_time = iris_end - iris_start
    wine_elapsed_time = wine_end - wine_start
    breast_cancer_elapsed_time = breast_cancer_end - breast_cancer_start
    glass_elapsed_time = glass_end - glass_start
    digits_elapsed_time = digits_end - digits_start
    rand_elapsed_time = rand_end - rand_start
    spiral_elapsed_time = spiral_end - spiral_start
    moons_elapsed_time = moons_end - moons_start
    circles_elapsed_time = circles_end - circles_start

    ############################### Get scores for CEKM ###############################
    print("########################## CEKM ##########################")
    print("Iris: " + str(iris_cekm_rand_score) +
          "\nWine: " + str(wine_cekm_rand_score) +
          "\nBreast: " + str(breast_cancer_cekm_rand_score) +
          "\nGlass: " + str(glass_cekm_rand_score) +
          "\nDigits: " + str(digits_cekm_rand_score) +
          "\nRand: " + str(rand_cekm_rand_score) +
          "\nSpiral: " + str(spiral_cekm_rand_score) +
          "\nMoons: " + str(moons_cekm_rand_score) +
          "\nCircles: " + str(circles_cekm_rand_score))

    print("########################## CEKM Time ##########################")
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
    alg = "CEKM "
    iris_plot = draw_data_2DNC(iris_set, np.asarray(iris_cekm_assignment, np.float), 3, alg + "Iris Dataset Results")
    rand_plot = draw_data_2DNC(rand_set, np.asarray(rand_cekm_assignment, np.float), 3, alg + "Rand Dataset Results")
    spiral_plot = draw_data_2DNC(spiral_set, np.asarray(spiral_cekm_assignment, np.float), 2, alg + "Spirals Dataset Results")
    moons_plot = draw_data_2DNC(moons_set, np.asarray(moons_cekm_assignment, np.float), 2, alg + "Moons Dataset Results")
    circles_plot = draw_data_2DNC(circles_set, np.asarray(circles_cekm_assignment, np.float), 2, alg + "Circles Dataset Results")
    plt.show()




if __name__ == "__main__": main()
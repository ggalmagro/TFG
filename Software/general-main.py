from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from CECM.CECM import CECM
from CKmeans.CKmeans import CKmeans
from LCVQE.LCVQE import LCVQE
from RDPmeans.RDPmeans import RDPmeans
from TVClust.TVClust import TVClust
from COPKmeans.COPKmeans import cop_kmeans
from functions import generate_data_2D, draw_data_2D, draw_data_2DNC, draw_const, add_constraints, twospirals
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
import gc
import pickle

def get_tvlust_input(m):
    output_m = -1 * np.ones(np.shape(m))
    checked = np.zeros(np.shape(m))
    output_m[m == 1] = 1
    output_m[m == -1] = 0
    checked[m != 0] = 1

    return output_m, checked

def get_lcvqe_input(m):
    constraint_number = np.count_nonzero(m - np.identity(np.shape(m)[0])) / 2 + np.shape(m)[0]
    list_const = np.zeros((int(constraint_number), 3), dtype=np.int)
    idx = 0
    for i in range(np.shape(m)[0]):
        for j in range(i + 1, np.shape(m)[0]):
            if m[i, j] != 0:
                list_const[idx, :] = [i, j, m[i, j]]
                idx += 1

    return list_const

def get_copkm_input(m):
    ml = []
    cl = []

    for i in range(np.shape(m)[0]):
        for j in range(i + 1, np.shape(m)[0]):
            if m[i, j] == 1:
                ml.append((i, j))
            if m[i, j] == -1:
                cl.append((i, j))

    return ml, cl

##### COMANDO: ctrl+d de sublime -> alt+j
# ax0, ax1, ax2 = draw_data_2D(set, labels, 2, [[0, 0], [0, 0], [0, 0]])
# plt.show()

def main():
    np.random.seed(43)
    random_state = 43
    const_percent = 0.25
    #Porcentaje para COP-Kmeans
    #const_percent = 0.1

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

    iris = []; wine = []; breast_cancer = []; glass = []; digits = []
    gc.collect()

    rand_set, rand_labels = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 100)
    spiral_set, spiral_labels = twospirals(150)
    spiral_set += 15
    moons_set, moons_labels = datasets.make_moons(300, .5, .05, random_state)
    moons_set += 1.5
    circles_set, circles_labels = datasets.make_circles(300, .5, .05, random_state, .3)
    circles_set += 1.5

    # iris_const = np.identity(len(iris_labels))
    # iris_const = add_constraints(iris_set, iris_labels, iris_const, int(len(iris_labels) * const_percent), 0, 1)
    #
    # wine_const = np.identity(len(wine_labels))
    # wine_const = add_constraints(wine_set, wine_labels, wine_const, int(len(wine_labels) * const_percent), 0, 1)
    #
    # breast_cancer_const = np.identity(len(breast_cancer_labels))
    # breast_cancer_const = add_constraints(breast_cancer_set, breast_cancer_labels, breast_cancer_const, int(len(breast_cancer_labels) * const_percent), 0, 1)
    #
    # glass_const = np.identity(len(glass_labels))
    # glass_const = add_constraints(glass_set, glass_labels, glass_const, int(len(glass_labels) * const_percent), 0, 1)
    #
    # digits_const = np.identity(len(digits_labels))
    # digits_const = add_constraints(digits_set, digits_labels, digits_const, int(len(digits_labels) * const_percent), 0, 1)
    #
    # rand_const = np.identity(len(rand_labels))
    # rand_const = add_constraints(rand_set, rand_labels, rand_const, int(len(rand_labels) * const_percent), 0, 1)
    #
    # spiral_const = np.identity(len(spiral_labels))
    # spiral_const = add_constraints(spiral_set, spiral_labels, spiral_const, int(len(spiral_labels) * const_percent), 0, 1)
    #
    # moons_const = np.identity(len(moons_labels))
    # moons_const = add_constraints(moons_set, moons_labels, moons_const, int(len(moons_labels) * const_percent), 0, 1)
    #
    # circles_const = np.identity(len(circles_labels))
    # circles_const = add_constraints(circles_set, circles_labels, circles_const, int(len(circles_labels) * const_percent), 0, 1)

    #np.savetxt('breast.txt', breast_cancer_set)

    # iris_labels_file = open("results/trueLabels/iris_labels.pkl", "wb")
    # pickle.dump(iris_labels, iris_labels_file)
    # iris_labels_file.close()
    #
    # wine_labels_file = open("results/trueLabels/wine_labels.pkl", "wb")
    # pickle.dump(wine_labels, wine_labels_file)
    # wine_labels_file.close()
    #
    # breast_cancer_labels_file = open("results/trueLabels/breast_cancer_labels.pkl", "wb")
    # pickle.dump(breast_cancer_labels, breast_cancer_labels_file)
    # breast_cancer_labels_file.close()
    #
    # glass_labels_file = open("results/trueLabels/glass_labels.pkl", "wb")
    # pickle.dump(glass_labels, glass_labels_file)
    # glass_labels_file.close()
    #
    # digits_labels_file = open("results/trueLabels/digits_labels.pkl", "wb")
    # pickle.dump(digits_labels, digits_labels_file)
    # digits_labels_file.close()
    #
    # rand_labels_file = open("results/trueLabels/rand_labels.pkl", "wb")
    # pickle.dump(rand_labels, rand_labels_file)
    # rand_labels_file.close()
    #
    # spiral_labels_file = open("results/trueLabels/spiral_labels.pkl", "wb")
    # pickle.dump(spiral_labels, spiral_labels_file)
    # spiral_labels_file.close()
    #
    # moons_labels_file = open("results/trueLabels/moons_labels.pkl", "wb")
    # pickle.dump(moons_labels, moons_labels_file)
    # moons_labels_file.close()
    #
    # circles_labels_file = open("results/trueLabels/circles_labels.pkl", "wb")
    # pickle.dump(circles_labels, circles_labels_file)
    # circles_labels_file.close()

    iris_centroid = []
    wine_centroid = []
    breast_cancer_centroid = []
    glass_centroid = []
    digits_centroid = []
    rand_centroid = []
    spiral_centroid = []
    moons_centroid = []
    circles_centroid = []

    ########################## Constrained K-means ############################
    # print ("COP-Kmeans")
    # iris_ml, iris_cl = get_copkm_input(iris_const)
    # wine_ml, wine_cl = get_copkm_input(wine_const)
    # breast_cancer_ml, breast_cancer_cl = get_copkm_input(breast_cancer_const)
    # glass_ml, glass_cl = get_copkm_input(glass_const)
    # digits_ml, digits_cl = get_copkm_input(digits_const)
    # rand_ml, rand_cl = get_copkm_input(rand_const)
    # spiral_ml, spiral_cl = get_copkm_input(spiral_const)
    # moons_ml, moons_cl = get_copkm_input(moons_const)
    # circles_ml, circles_cl = get_copkm_input(circles_const)
    #
    # iris_ckm_assignment, iris_centroid = cop_kmeans(iris_set, 3, iris_ml, iris_cl)
    # wine_ckm_assignment, wine_centroid = cop_kmeans(wine_set, 3, wine_ml, wine_cl)
    # glass_ckm_assignment, glass_centroid = cop_kmeans(glass_set, 6, glass_ml, glass_cl)
    # breast_cancer_ckm_assignment, breast_cancer_centroid = cop_kmeans(breast_cancer_set, 2, breast_cancer_ml, breast_cancer_cl)
    # digits_ckm_assignment, digits_centroid = cop_kmeans(digits_set, 10, digits_ml, digits_cl)
    # rand_ckm_assignment, rand_centroid = cop_kmeans(rand_set, 3, rand_ml, rand_cl)
    # spiral_ckm_assignment, spiral_centroid = cop_kmeans(spiral_set, 2, spiral_ml, spiral_cl)
    # moons_ckm_assignment, moons_centroid = cop_kmeans(moons_set, 2, moons_ml, moons_cl)
    # circles_ckm_assignment, circles_centroid = cop_kmeans(circles_set, 2, circles_ml, circles_cl)
    #
    # # Saving Results
    # iris_centroid_file = open("results/CKmeans/iris_centroid.pkl", "wb")
    # pickle.dump(iris_centroid, iris_centroid_file)
    # iris_centroid_file.close()
    # iris_ckm_assignment_file = open("results/CKmeans/iris_ckm_assignment.pkl", "wb")
    # pickle.dump(iris_ckm_assignment, iris_ckm_assignment_file)
    # iris_ckm_assignment_file.close()
    #
    # #Saving Results
    # wine_centroid_file = open("results/CKmeans/wine_centroid.pkl", "wb")
    # pickle.dump(wine_centroid, wine_centroid_file)
    # wine_centroid_file.close()
    # wine_ckm_assignment_file = open("results/CKmeans/wine_ckm_assignment.pkl", "wb")
    # pickle.dump(wine_ckm_assignment, wine_ckm_assignment_file)
    # wine_ckm_assignment_file.close()
    #
    # # Saving Results
    # breast_cancer_centroid_file = open("results/CKmeans/breast_cancer_centroid.pkl", "wb")
    # pickle.dump(breast_cancer_centroid, breast_cancer_centroid_file)
    # breast_cancer_centroid_file.close()
    # breast_cancer_ckm_assignment_file = open("results/CKmeans/breast_cancer_ckm_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_ckm_assignment, breast_cancer_ckm_assignment_file)
    # breast_cancer_ckm_assignment_file.close()
    #
    # # Saving Results
    # glass_centroid_file = open("results/CKmeans/glass_centroid.pkl", "wb")
    # pickle.dump(glass_centroid, glass_centroid_file)
    # glass_centroid_file.close()
    # glass_ckm_assignment_file = open("results/CKmeans/glass_ckm_assignment.pkl", "wb")
    # pickle.dump(glass_ckm_assignment, glass_ckm_assignment_file)
    # glass_ckm_assignment_file.close()
    #
    # # Saving Results
    # digits_centroid_file = open("results/CKmeans/digits_centroid.pkl", "wb")
    # pickle.dump(digits_centroid, digits_centroid_file)
    # digits_centroid_file.close()
    # digits_ckm_assignment_file = open("results/CKmeans/digits_ckm_assignment.pkl", "wb")
    # pickle.dump(digits_ckm_assignment, digits_ckm_assignment_file)
    # digits_ckm_assignment_file.close()
    #
    #
    # # Saving Results
    # rand_centroid_file = open("results/CKmeans/rand_centroid.pkl", "wb")
    # pickle.dump(rand_centroid, rand_centroid_file)
    # rand_centroid_file.close()
    # rand_ckm_assignment_file = open("results/CKmeans/rand_ckm_assignment.pkl", "wb")
    # pickle.dump(rand_ckm_assignment, rand_ckm_assignment_file)
    # rand_ckm_assignment_file.close()
    #
    #
    # # Saving Results
    # spiral_centroid_file = open("results/CKmeans/spiral_centroid.pkl", "wb")
    # pickle.dump(spiral_centroid, spiral_centroid_file)
    # spiral_centroid_file.close()
    # spiral_ckm_assignment_file = open("results/CKmeans/spiral_ckm_assignment.pkl", "wb")
    # pickle.dump(spiral_ckm_assignment, spiral_ckm_assignment_file)
    # spiral_ckm_assignment_file.close()
    #
    #
    # # Saving Results
    # moons_centroid_file = open("results/CKmeans/moons_centroid.pkl", "wb")
    # pickle.dump(moons_centroid, moons_centroid_file)
    # moons_centroid_file.close()
    # moons_ckm_assignment_file = open("results/CKmeans/moons_ckm_assignment.pkl", "wb")
    # pickle.dump(moons_ckm_assignment, moons_ckm_assignment_file)
    # moons_ckm_assignment_file.close()
    #
    # # Saving Results
    # circles_centroid_file = open("results/CKmeans/circles_centroid.pkl", "wb")
    # pickle.dump(circles_centroid, circles_centroid_file)
    # circles_centroid_file.close()
    # circles_ckm_assignment_file = open("results/CKmeans/circles_ckm_assignment.pkl", "wb")
    # pickle.dump(circles_ckm_assignment, circles_ckm_assignment_file)
    # circles_ckm_assignment_file.close()


    ########################### Constrained Evidential K-means ############################
    # print("CECM")
    # iris_cecm_assignment = CECM(iris_set, 3, iris_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # # wine_cecm_assignment = CECM(wine_set, 3, wine_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # # breast_cancer_cecm_assignment = CECM(breast_cancer_set, 2, breast_cancer_const, alpha=1, rho2=1000, distance=0, bal= 0, init=1)
    # # glass_cecm_assignment = CECM(glass_set, 6, glass_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # # digits_cecm_assignment = CECM(digits_set, 10, digits_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    #
    # rand_cecm_assignment = CECM(rand_set, 3, rand_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # spiral_cecm_assignment = CECM(spiral_set, 2, spiral_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # moons_cecm_assignment = CECM(moons_set, 2, moons_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    # circles_cecm_assignment = CECM(circles_set, 2, circles_const, alpha=1, rho2=1000, distance=0, bal=0, init=1)
    #
    # # Saving Results
    # iris_cecm_assignment_file = open("results/CECM/iris_cecm_assignment.pkl", "wb")
    # pickle.dump(iris_cecm_assignment, iris_cecm_assignment_file)
    # iris_cecm_assignment_file.close()
    #
    # wine_cecm_assignment_file = open("results/CECM/wine_cecm_assignment.pkl", "wb")
    # pickle.dump(wine_cecm_assignment, wine_cecm_assignment_file)
    # wine_cecm_assignment_file.close()
    #
    # breast_cancer_cecm_assignment_file = open("results/CECM/breast_cancer_cecm_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_cecm_assignment, breast_cancer_cecm_assignment_file)
    # breast_cancer_cecm_assignment_file.close()
    #
    # # glass_cecm_assignment_file = open("results/CECM/glass_cecm_assignment.pkl", "wb")
    # # pickle.dump(glass_cecm_assignment, glass_cecm_assignment_file)
    # # glass_cecm_assignment_file.close()
    # #
    # # digits_cecm_assignment_file = open("results/CECM/digits_cecm_assignment.pkl", "wb")
    # # pickle.dump(digits_cecm_assignment, digits_cecm_assignment_file)
    # # digits_cecm_assignment_file.close()
    #
    # rand_cecm_assignment_file = open("results/CECM/rand_cecm_assignment.pkl", "wb")
    # pickle.dump(rand_cecm_assignment, rand_cecm_assignment_file)
    # rand_cecm_assignment_file.close()
    #
    # spiral_cecm_assignment_file = open("results/CECM/spiral_cecm_assignment.pkl", "wb")
    # pickle.dump(spiral_cecm_assignment, spiral_cecm_assignment_file)
    # spiral_cecm_assignment_file.close()
    #
    # moons_cecm_assignment_file = open("results/CECM/moons_cecm_assignment.pkl", "wb")
    # pickle.dump(moons_cecm_assignment, moons_cecm_assignment_file)
    # moons_cecm_assignment_file.close()
    #
    # circles_cecm_assignment_file = open("results/CECM/circles_cecm_assignment.pkl", "wb")
    # pickle.dump(circles_cecm_assignment, circles_cecm_assignment_file)
    # circles_cecm_assignment_file.close()
    #
    # ########################### TV- Clustering ############################
    # print("TV-clust")
    # iris_const2, iris_checked = get_tvlust_input(iris_const)
    # wine_const2, wine_checked = get_tvlust_input(wine_const)
    # breast_cancer_const2, breast_cancer_checked = get_tvlust_input(breast_cancer_const)
    # glass_const2, glass_checked = get_tvlust_input(glass_const)
    # digits_const2, digits_checked = get_tvlust_input(digits_const)
    #
    # iris_tvclust_assignment = TVClust(iris_set, iris_const2, iris_checked, 3)
    # wine_tvclust_assignment = TVClust(wine_set, wine_const2, wine_checked, 3)
    # breast_cancer_tvclust_assignment = TVClust(breast_cancer_set, breast_cancer_const2, breast_cancer_checked, 2)
    # glass_tvclust_assignment = TVClust(glass_set, glass_const2, glass_checked, 6)
    # digits_tvclust_assignment = TVClust(digits_set, digits_const2, digits_checked, 10)
    #
    # rand_const2, rand_checked = get_tvlust_input(rand_const)
    # spiral_const2, spiral_checked = get_tvlust_input(spiral_const)
    # moons_const2, moons_checked = get_tvlust_input(moons_const)
    # circles_const2, circles_checked = get_tvlust_input(circles_const)
    #
    # rand_tvclust_assignment = TVClust(rand_set, rand_const2, rand_checked, 3)
    # spiral_tvclust_assignment = TVClust(spiral_set, spiral_const2, spiral_checked, 2)
    # moons_tvclust_assignment = TVClust(moons_set, moons_const2, moons_checked, 2)
    # circles_tvclust_assignment = TVClust(circles_set, circles_const2, circles_checked, 2)
    #
    # # Saving Results
    # iris_tvclust_assignment_file = open("results/TVClust/iris_tvclust_assignment.pkl", "wb")
    # pickle.dump(iris_tvclust_assignment, iris_tvclust_assignment_file)
    # iris_tvclust_assignment_file.close()
    #
    # wine_tvclust_assignment_file = open("results/TVClust/wine_tvclust_assignment.pkl", "wb")
    # pickle.dump(wine_tvclust_assignment, wine_tvclust_assignment_file)
    # wine_tvclust_assignment_file.close()
    #
    # breast_cancer_tvclust_assignment_file = open("results/TVClust/breast_cancer_tvclust_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_tvclust_assignment, breast_cancer_tvclust_assignment_file)
    # breast_cancer_tvclust_assignment_file.close()
    #
    # glass_tvclust_assignment_file = open("results/TVClust/glass_tvclust_assignment.pkl", "wb")
    # pickle.dump(glass_tvclust_assignment, glass_tvclust_assignment_file)
    # glass_tvclust_assignment_file.close()
    #
    # digits_tvclust_assignment_file = open("results/TVClust/digits_tvclust_assignment.pkl", "wb")
    # pickle.dump(digits_tvclust_assignment, digits_tvclust_assignment_file)
    # digits_tvclust_assignment_file.close()
    #
    # rand_tvclust_assignment_file = open("results/TVClust/rand_tvclust_assignment.pkl", "wb")
    # pickle.dump(rand_tvclust_assignment, rand_tvclust_assignment_file)
    # rand_tvclust_assignment_file.close()
    #
    # spiral_tvclust_assignment_file = open("results/TVClust/spiral_tvclust_assignment.pkl", "wb")
    # pickle.dump(spiral_tvclust_assignment, spiral_tvclust_assignment_file)
    # spiral_tvclust_assignment_file.close()
    #
    # moons_tvclust_assignment_file = open("results/TVClust/moons_tvclust_assignment.pkl", "wb")
    # pickle.dump(moons_tvclust_assignment, moons_tvclust_assignment_file)
    # moons_tvclust_assignment_file.close()
    #
    # circles_tvclust_assignment_file = open("results/TVClust/circles_tvclust_assignment.pkl", "wb")
    # pickle.dump(circles_tvclust_assignment, circles_tvclust_assignment_file)
    # circles_tvclust_assignment_file.close()
    #
    # ########################## RDP - means ############################
    # print("RDP-means")
    # T = np.mean(iris_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(iris_set)[0], 1) - iris_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # iris_rdpm_assignment = RDPmeans(iris_set, 2, iris_const, 1, 0.1, 20000)[2]
    # T = np.mean(wine_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(wine_set)[0], 1) - wine_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # wine_rdpm_assignment = RDPmeans(wine_set, lamb_arr[2], wine_const, 1, 0.1, 20000)[2]
    # T = np.mean(breast_cancer_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(breast_cancer_set)[0], 1) - breast_cancer_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # breast_cancer_rdpm_assignment = RDPmeans(breast_cancer_set, lamb_arr[1], breast_cancer_const, 1, 0.1, 20000)[2]
    # T = np.mean(glass_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(glass_set)[0], 1) - glass_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # glass_rdpm_assignment = RDPmeans(glass_set, lamb_arr[5], glass_const, 1, 0.1, 20000)[2]
    # T = np.mean(digits_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(digits_set)[0], 1) - digits_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # digits_rdpm_assignment = RDPmeans(digits_set, lamb_arr[9], digits_const, 1, 0.1, 20000)[2]

    # T = np.mean(rand_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(rand_set)[0], 1) - rand_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # rand_rdpm_assignment,  rand_rdpm_nbc = RDPmeans(rand_set, 4, rand_const, 1, 0.1, 20000)
    # T = np.mean(spiral_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(spiral_set)[0], 1) - spiral_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # spiral_rdpm_assignment, spiral_rdpm_nbc = RDPmeans(spiral_set, lamb_arr[1], spiral_const, 1, 0.1, 20000)
    # T = np.mean(moons_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(moons_set)[0], 1) - moons_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # moons_rdpm_assignment, moons_rdpm_nbc = RDPmeans(moons_set, 2, moons_const, 1, 0.1, 20000)
    # T = np.mean(circles_set, 0)
    # lamb_arr = np.array(np.sqrt(np.sum((np.matlib.repmat(T, np.shape(circles_set)[0], 1) - circles_set) ** 2, 1)))
    # lamb_arr[::-1].sort()
    # circles_rdpm_assignment, circles_rdpm_nbc = RDPmeans(circles_set, 1.5, circles_const, 1, 0.1, 20000)
    #
    # #Saving Results
    # iris_rdpm_assignment_file = open("results/RDPmeans/iris_rdpm_assignment.pkl", "wb")
    # pickle.dump(iris_rdpm_assignment, iris_rdpm_assignment_file)
    # iris_rdpm_assignment_file.close()
    #
    # wine_rdpm_assignment_file = open("results/RDPmeans/wine_rdpm_assignment.pkl", "wb")
    # pickle.dump(wine_rdpm_assignment, wine_rdpm_assignment_file)
    # wine_rdpm_assignment_file.close()
    #
    # breast_cancer_rdpm_assignment_file = open("results/RDPmeans/breast_cancer_rdpm_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_rdpm_assignment, breast_cancer_rdpm_assignment_file)
    # breast_cancer_rdpm_assignment_file.close()
    #
    # glass_rdpm_assignment_file = open("results/RDPmeans/glass_rdpm_assignment.pkl", "wb")
    # pickle.dump(glass_rdpm_assignment, glass_rdpm_assignment_file)
    # glass_rdpm_assignment_file.close()
    #
    # digits_rdpm_assignment_file = open("results/RDPmeans/digits_rdpm_assignment.pkl", "wb")
    # pickle.dump(digits_rdpm_assignment, digits_rdpm_assignment_file)
    # digits_rdpm_assignment_file.close()
    #
    # rand_rdpm_assignment_file = open("results/RDPmeans/rand_rdpm_assignment.pkl", "wb")
    # pickle.dump(rand_rdpm_assignment, rand_rdpm_assignment_file)
    # rand_rdpm_assignment_file.close()
    #
    # spiral_rdpm_assignment_file = open("results/RDPmeans/spiral_rdpm_assignment.pkl", "wb")
    # pickle.dump(spiral_rdpm_assignment, spiral_rdpm_assignment_file)
    # spiral_rdpm_assignment_file.close()
    #
    # moons_rdpm_assignment_file = open("results/RDPmeans/moons_rdpm_assignment.pkl", "wb")
    # pickle.dump(moons_rdpm_assignment, moons_rdpm_assignment_file)
    # moons_rdpm_assignment_file.close()
    #
    # circles_rdpm_assignment_file = open("results/RDPmeans/circles_rdpm_assignment.pkl", "wb")
    # pickle.dump(circles_rdpm_assignment, circles_rdpm_assignment_file)
    # circles_rdpm_assignment_file.close()
    #
    # ########################### LCVQE ############################
    # print("LCVQE")
    # iris_const_list = get_lcvqe_input(iris_const)
    # wine_const_list = get_lcvqe_input(wine_const)
    # breast_cancer_const_list = get_lcvqe_input(breast_cancer_const)
    # glass_const_list = get_lcvqe_input(glass_const)
    # digits_const_list = get_lcvqe_input(digits_const)
    #
    # #Los centroides no se usan dentro del algoritmo
    # iris_lcvqe_assignment = LCVQE(iris_set, 3, iris_const_list, iris_centroid)
    # wine_lcvqe_assignment = LCVQE(wine_set, 3, wine_const_list, wine_centroid)
    # breast_cancer_lcvqe_assignment = LCVQE(breast_cancer_set, 2, breast_cancer_const_list, breast_cancer_centroid)
    # glass_lcvqe_assignment = LCVQE(glass_set, 6, glass_const_list, glass_centroid)
    # digits_lcvqe_assignment = LCVQE(digits_set, 10, digits_const_list, digits_centroid)

    # rand_const_list = get_lcvqe_input(rand_const)
    # spiral_const_list = get_lcvqe_input(spiral_const)
    # moons_const_list = get_lcvqe_input(moons_const)
    # circles_const_list = get_lcvqe_input(circles_const)
    #
    # rand_lcvqe_assignment = LCVQE(rand_set, 3, rand_const_list, rand_centroid)
    # spiral_lcvqe_assignment = LCVQE(spiral_set, 2, spiral_const_list, spiral_centroid)
    # moons_lcvqe_assignment = LCVQE(moons_set, 2, moons_const_list, moons_centroid)
    # circles_lcvqe_assignment = LCVQE(circles_set, 2, circles_const_list, circles_centroid)
    #
    # iris_lcvqe_assignment_file = open("results/LCVQE/iris_lcvqe_assignment.pkl", "wb")
    # pickle.dump(iris_lcvqe_assignment, iris_lcvqe_assignment_file)
    # iris_lcvqe_assignment_file.close()
    #
    # wine_lcvqe_assignment_file = open("results/LCVQE/wine_lcvqe_assignment.pkl", "wb")
    # pickle.dump(wine_lcvqe_assignment, wine_lcvqe_assignment_file)
    # wine_lcvqe_assignment_file.close()
    #
    # breast_cancer_lcvqe_assignment_file = open("results/LCVQE/breast_cancer_lcvqe_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_lcvqe_assignment, breast_cancer_lcvqe_assignment_file)
    # breast_cancer_lcvqe_assignment_file.close()
    #
    # glass_lcvqe_assignment_file = open("results/LCVQE/glass_lcvqe_assignment.pkl", "wb")
    # pickle.dump(glass_lcvqe_assignment, glass_lcvqe_assignment_file)
    # glass_lcvqe_assignment_file.close()
    #
    # digits_lcvqe_assignment_file = open("results/LCVQE/digits_lcvqe_assignment.pkl", "wb")
    # pickle.dump(digits_lcvqe_assignment, digits_lcvqe_assignment_file)
    # digits_lcvqe_assignment_file.close()
    #
    # rand_lcvqe_assignment_file = open("results/LCVQE/rand_lcvqe_assignment.pkl", "wb")
    # pickle.dump(rand_lcvqe_assignment, rand_lcvqe_assignment_file)
    # rand_lcvqe_assignment_file.close()
    #
    # spiral_lcvqe_assignment_file = open("results/LCVQE/spiral_lcvqe_assignment.pkl", "wb")
    # pickle.dump(spiral_lcvqe_assignment, spiral_lcvqe_assignment_file)
    # spiral_lcvqe_assignment_file.close()
    #
    # moons_lcvqe_assignment_file = open("results/LCVQE/moons_lcvqe_assignment.pkl", "wb")
    # pickle.dump(moons_lcvqe_assignment, moons_lcvqe_assignment_file)
    # moons_lcvqe_assignment_file.close()
    #
    # circles_lcvqe_assignment_file = open("results/LCVQE/circles_lcvqe_assignment.pkl", "wb")
    # pickle.dump(circles_lcvqe_assignment, circles_lcvqe_assignment_file)
    # circles_lcvqe_assignment_file.close()
    #
    # ############################ K-means ############################
    # print("K-means")
    # iris_km_assignment = KMeans(init="random", n_clusters=3).fit(iris_set).labels_
    # wine_km_assignment = KMeans(init="random", n_clusters=3).fit(wine_set).labels_
    # breast_cancer_km_assignment = KMeans(init="random", n_clusters=2).fit(breast_cancer_set).labels_
    # glass_km_assignment = KMeans(init="random", n_clusters=6).fit(glass_set).labels_
    # digits_km_assignment = KMeans(init="random", n_clusters=10).fit(digits_set).labels_
    # rand_km_assignment = KMeans(init="random", n_clusters=2).fit(rand_set).labels_
    # spiral_km_assignment = KMeans(init="random", n_clusters=2).fit(spiral_set).labels_
    # moons_km_assignment = KMeans(init="random", n_clusters=2).fit(moons_set).labels_
    # circles_km_assignment = KMeans(init="random", n_clusters=2).fit(circles_set).labels_
    #
    # #Saving Results
    # iris_km_assignment_file = open("results/Kmeans/iris_km_assignment.pkl", "wb")
    # pickle.dump(iris_km_assignment, iris_km_assignment_file)
    # iris_km_assignment_file.close()
    #
    # wine_km_assignment_file = open("results/Kmeans/wine_km_assignment.pkl", "wb")
    # pickle.dump(wine_km_assignment, wine_km_assignment_file)
    # wine_km_assignment_file.close()
    #
    # breast_cancer_km_assignment_file = open("results/Kmeans/breast_cancer_km_assignment.pkl", "wb")
    # pickle.dump(breast_cancer_km_assignment, breast_cancer_km_assignment_file)
    # breast_cancer_km_assignment_file.close()
    #
    # glass_km_assignment_file = open("results/Kmeans/glass_km_assignment.pkl", "wb")
    # pickle.dump(glass_km_assignment, glass_km_assignment_file)
    # glass_km_assignment_file.close()
    #
    # digits_km_assignment_file = open("results/Kmeans/digits_km_assignment.pkl", "wb")
    # pickle.dump(digits_km_assignment, digits_km_assignment_file)
    # digits_km_assignment_file.close()
    #
    # rand_km_assignment_file = open("results/Kmeans/rand_km_assignment.pkl", "wb")
    # pickle.dump(rand_km_assignment, rand_km_assignment_file)
    # rand_km_assignment_file.close()
    #
    # spiral_km_assignment_file = open("results/Kmeans/spiral_km_assignment.pkl", "wb")
    # pickle.dump(spiral_km_assignment, spiral_km_assignment_file)
    # spiral_km_assignment_file.close()
    #
    # moons_km_assignment_file = open("results/Kmeans/moons_km_assignment.pkl", "wb")
    # pickle.dump(moons_km_assignment, moons_km_assignment_file)
    # moons_km_assignment_file.close()
    #
    # circles_km_assignment_file = open("results/Kmeans/circles_km_assignment.pkl", "wb")
    # pickle.dump(circles_km_assignment, circles_km_assignment_file)
    # circles_km_assignment_file.close()

    ########################### Drawing results ###########################
    # ax0 = draw_data_2DNC(iris_set, np.asarray(iris_ckm_assignment, np.float), 3, "K-means Iris Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(rand_set, np.asarray(rand_ckm_assignment, np.float), 3, "K-means 3 Clusters Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(spiral_set, np.asarray(spiral_ckm_assignment, np.float), 2, "K-means Spirals Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(moons_set, np.asarray(moons_ckm_assignment, np.float), 2, "K-means Moons Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(circles_set, np.asarray(circles_ckm_assignment, np.float), 2, "K-means Circles Dataset")
    # plt.show()

    # ax0 = draw_data_2DNC(iris_set, np.asarray(iris_cecm_assignment, np.float), 3, "K-means Iris Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(rand_set, np.asarray(rand_cecm_assignment, np.float), 3, "K-means 3 Clusters Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(spiral_set, np.asarray(spiral_cecm_assignment, np.float), 2, "K-means Spirals Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(moons_set, np.asarray(moons_cecm_assignment, np.float), 2, "K-means Moons Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(circles_set, np.asarray(circles_cecm_assignment, np.float), 2, "K-means Circles Dataset")
    # plt.show()

    # ax0 = draw_data_2DNC(iris_set, np.asarray(iris_tvclust_assignment, np.float), 3, "K-means Iris Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(rand_set, np.asarray(rand_tvclust_assignment, np.float), 3, "K-means 3 Clusters Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(spiral_set, np.asarray(spiral_tvclust_assignment, np.float), 2, "K-means Spirals Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(moons_set, np.asarray(moons_tvclust_assignment, np.float), 2, "K-means Moons Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(circles_set, np.asarray(circles_tvclust_assignment, np.float), 2, "K-means Circles Dataset")
    # plt.show()

    # ax0 = draw_data_2DNC(iris_set, np.asarray(iris_rdpm_assignment, np.float), 3, "K-means Iris Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(rand_set, np.asarray(rand_rdpm_assignment, np.float), rand_rdpm_nbc, "K-means 3 Clusters Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(spiral_set, np.asarray(spiral_rdpm_assignment, np.float), spiral_rdpm_nbc, "K-means Spirals Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(moons_set, np.asarray(moons_rdpm_assignment, np.float), moons_rdpm_nbc, "K-means Moons Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(circles_set, np.asarray(circles_rdpm_assignment, np.float), circles_rdpm_nbc, "K-means Circles Dataset")
    # plt.show()

    # ax0 = draw_data_2DNC(iris_set, np.asarray(iris_lcvqe_assignment, np.float), 3, "K-means Iris Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(rand_set, np.asarray(rand_lcvqe_assignment, np.float), 3, "K-means 3 Clusters Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(spiral_set, np.asarray(spiral_lcvqe_assignment, np.float), 2, "K-means Spirals Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(moons_set, np.asarray(moons_lcvqe_assignment, np.float), 2, "K-means Moons Dataset")
    # plt.show()
    # ax0 = draw_data_2DNC(circles_set, np.asarray(circles_lcvqe_assignment, np.float), 2, "K-means Circles Dataset")
    # plt.show()

if __name__ == "__main__": main()
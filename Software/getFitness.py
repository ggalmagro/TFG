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
from functions import generate_data_2D, draw_data_2D, draw_const, add_constraints, twospirals
from sklearn.datasets import fetch_mldata
from sklearn.metrics import adjusted_rand_score
import gc
import pickle


def main():

    # iris, wine, breast_cancer, glass, digits, rand, spiral, moons, circles
    source = "results"
    ############################### Load K-means results ###############################
    iris_km_assignment_file = open(source + "/Kmeans/iris_km_assignment.pkl", "rb")
    iris_km_assignment = pickle.load(iris_km_assignment_file)
    iris_km_assignment_file.close()

    wine_km_assignment_file = open(source + "/Kmeans/wine_km_assignment.pkl", "rb")
    wine_km_assignment = pickle.load(wine_km_assignment_file)
    wine_km_assignment_file.close()

    breast_cancer_km_assignment_file = open(source + "/Kmeans/breast_cancer_km_assignment.pkl", "rb")
    breast_cancer_km_assignment = pickle.load(breast_cancer_km_assignment_file)
    breast_cancer_km_assignment_file.close()

    glass_km_assignment_file = open(source + "/Kmeans/glass_km_assignment.pkl", "rb")
    glass_km_assignment = pickle.load(glass_km_assignment_file)
    glass_km_assignment_file.close()

    digits_km_assignment_file = open(source + "/Kmeans/digits_km_assignment.pkl", "rb")
    digits_km_assignment = pickle.load(digits_km_assignment_file)
    digits_km_assignment_file.close()

    rand_km_assignment_file = open(source + "/Kmeans/rand_km_assignment.pkl", "rb")
    rand_km_assignment = pickle.load(rand_km_assignment_file)
    rand_km_assignment_file.close()

    spiral_km_assignment_file = open(source + "/Kmeans/spiral_km_assignment.pkl", "rb")
    spiral_km_assignment = pickle.load(spiral_km_assignment_file)
    spiral_km_assignment_file.close()

    moons_km_assignment_file = open(source + "/Kmeans/moons_km_assignment.pkl", "rb")
    moons_km_assignment = pickle.load(moons_km_assignment_file)
    moons_km_assignment_file.close()

    circles_km_assignment_file = open(source + "/Kmeans/circles_km_assignment.pkl", "rb")
    circles_km_assignment = pickle.load(circles_km_assignment_file)
    circles_km_assignment_file.close()

    ############################### Load CMK results ###############################
    iris_ckm_assignment_file = open(source + "/CKmeans/iris_ckm_assignment.pkl", "rb")
    iris_ckm_assignment = pickle.load(iris_ckm_assignment_file)
    iris_ckm_assignment_file.close()

    wine_ckm_assignment_file = open(source + "/CKmeans/wine_ckm_assignment.pkl", "rb")
    wine_ckm_assignment = pickle.load(wine_ckm_assignment_file)
    wine_ckm_assignment_file.close()

    breast_cancer_ckm_assignment_file = open(source + "/CKmeans/breast_cancer_ckm_assignment.pkl", "rb")
    breast_cancer_ckm_assignment = pickle.load(breast_cancer_ckm_assignment_file)
    breast_cancer_ckm_assignment_file.close()

    glass_ckm_assignment_file = open(source + "/CKmeans/glass_ckm_assignment.pkl", "rb")
    glass_ckm_assignment = pickle.load(glass_ckm_assignment_file)
    glass_ckm_assignment_file.close()

    digits_ckm_assignment_file = open(source + "/CKmeans/digits_ckm_assignment.pkl", "rb")
    digits_ckm_assignment = pickle.load(digits_ckm_assignment_file)
    digits_ckm_assignment_file.close()

    rand_ckm_assignment_file = open(source + "/CKmeans/rand_ckm_assignment.pkl", "rb")
    rand_ckm_assignment = pickle.load(rand_ckm_assignment_file)
    rand_ckm_assignment_file.close()

    spiral_ckm_assignment_file = open(source + "/CKmeans/spiral_ckm_assignment.pkl", "rb")
    spiral_ckm_assignment = pickle.load(spiral_ckm_assignment_file)
    spiral_ckm_assignment_file.close()

    moons_ckm_assignment_file = open(source + "/CKmeans/moons_ckm_assignment.pkl", "rb")
    moons_ckm_assignment = pickle.load(moons_ckm_assignment_file)
    moons_ckm_assignment_file.close()

    circles_ckm_assignment_file = open(source + "/CKmeans/circles_ckm_assignment.pkl", "rb")
    circles_ckm_assignment = pickle.load(circles_ckm_assignment_file)
    circles_ckm_assignment_file.close()

    ############################### Load CECM results ###############################
    iris_cecm_assignment_file = open(source + "/CECM/iris_cecm_assignment.pkl", "rb")
    iris_cecm_assignment = pickle.load(iris_cecm_assignment_file)
    iris_cecm_assignment_file.close()

    wine_cecm_assignment_file = open(source + "/CECM/wine_cecm_assignment.pkl", "rb")
    wine_cecm_assignment = pickle.load(wine_cecm_assignment_file)
    wine_cecm_assignment_file.close()

    breast_cancer_cecm_assignment_file = open(source + "/CECM/breast_cancer_cecm_assignment.pkl", "rb")
    breast_cancer_cecm_assignment = pickle.load(breast_cancer_cecm_assignment_file)
    breast_cancer_cecm_assignment_file.close()

    # glass_cecm_assignment_file = open(source + "/CECM/glass_cecm_assignment.pkl", "rb")
    # glass_cecm_assignment = pickle.load(glass_cecm_assignment_file)
    # glass_cecm_assignment_file.close()
    #
    # digits_cecm_assignment_file = open(source + "/CECM/digits_cecm_assignment.pkl", "rb")
    # digits_cecm_assignment = pickle.load(digits_cecm_assignment_file)
    # digits_cecm_assignment_file.close()

    rand_cecm_assignment_file = open(source + "/CECM/rand_cecm_assignment.pkl", "rb")
    rand_cecm_assignment = pickle.load(rand_cecm_assignment_file)
    rand_cecm_assignment_file.close()

    spiral_cecm_assignment_file = open(source + "/CECM/spiral_cecm_assignment.pkl", "rb")
    spiral_cecm_assignment = pickle.load(spiral_cecm_assignment_file)
    spiral_cecm_assignment_file.close()

    moons_cecm_assignment_file = open(source + "/CECM/moons_cecm_assignment.pkl", "rb")
    moons_cecm_assignment = pickle.load(moons_cecm_assignment_file)
    moons_cecm_assignment_file.close()

    circles_cecm_assignment_file = open(source + "/CECM/circles_cecm_assignment.pkl", "rb")
    circles_cecm_assignment = pickle.load(circles_cecm_assignment_file)
    circles_cecm_assignment_file.close()

    ############################### Load TVClust results ###############################
    iris_tvclust_assignment_file = open(source + "/TVClust/iris_tvclust_assignment.pkl", "rb")
    iris_tvclust_assignment = pickle.load(iris_tvclust_assignment_file)
    iris_tvclust_assignment_file.close()

    wine_tvclust_assignment_file = open(source + "/TVClust/wine_tvclust_assignment.pkl", "rb")
    wine_tvclust_assignment = pickle.load(wine_tvclust_assignment_file)
    wine_tvclust_assignment_file.close()

    breast_cancer_tvclust_assignment_file = open(source + "/TVClust/breast_cancer_tvclust_assignment.pkl", "rb")
    breast_cancer_tvclust_assignment = pickle.load(breast_cancer_tvclust_assignment_file)
    breast_cancer_tvclust_assignment_file.close()

    glass_tvclust_assignment_file = open(source + "/TVClust/glass_tvclust_assignment.pkl", "rb")
    glass_tvclust_assignment = pickle.load(glass_tvclust_assignment_file)
    glass_tvclust_assignment_file.close()

    digits_tvclust_assignment_file = open(source + "/TVClust/digits_tvclust_assignment.pkl", "rb")
    digits_tvclust_assignment = pickle.load(digits_tvclust_assignment_file)
    digits_tvclust_assignment_file.close()

    rand_tvclust_assignment_file = open(source + "/TVClust/rand_tvclust_assignment.pkl", "rb")
    rand_tvclust_assignment = pickle.load(rand_tvclust_assignment_file)
    rand_tvclust_assignment_file.close()

    spiral_tvclust_assignment_file = open(source + "/TVClust/spiral_tvclust_assignment.pkl", "rb")
    spiral_tvclust_assignment = pickle.load(spiral_tvclust_assignment_file)
    spiral_tvclust_assignment_file.close()

    moons_tvclust_assignment_file = open(source + "/TVClust/moons_tvclust_assignment.pkl", "rb")
    moons_tvclust_assignment = pickle.load(moons_tvclust_assignment_file)
    moons_tvclust_assignment_file.close()

    circles_tvclust_assignment_file = open(source + "/TVClust/circles_tvclust_assignment.pkl", "rb")
    circles_tvclust_assignment = pickle.load(circles_tvclust_assignment_file)
    circles_tvclust_assignment_file.close()

    ############################### Load RDPM results ###############################
    iris_rdpm_assignment_file = open(source + "/RDPmeans/iris_rdpm_assignment.pkl", "rb")
    iris_rdpm_assignment = pickle.load(iris_rdpm_assignment_file)
    iris_rdpm_assignment_file.close()

    wine_rdpm_assignment_file = open(source + "/RDPmeans/wine_rdpm_assignment.pkl", "rb")
    wine_rdpm_assignment = pickle.load(wine_rdpm_assignment_file)
    wine_rdpm_assignment_file.close()

    breast_cancer_rdpm_assignment_file = open(source + "/RDPmeans/breast_cancer_rdpm_assignment.pkl", "rb")
    breast_cancer_rdpm_assignment = pickle.load(breast_cancer_rdpm_assignment_file)
    breast_cancer_rdpm_assignment_file.close()

    glass_rdpm_assignment_file = open(source + "/RDPmeans/glass_rdpm_assignment.pkl", "rb")
    glass_rdpm_assignment = pickle.load(glass_rdpm_assignment_file)
    glass_rdpm_assignment_file.close()

    digits_rdpm_assignment_file = open(source + "/RDPmeans/digits_rdpm_assignment.pkl", "rb")
    digits_rdpm_assignment = pickle.load(digits_rdpm_assignment_file)
    digits_rdpm_assignment_file.close()

    rand_rdpm_assignment_file = open(source + "/RDPmeans/rand_rdpm_assignment.pkl", "rb")
    rand_rdpm_assignment = pickle.load(rand_rdpm_assignment_file)
    rand_rdpm_assignment_file.close()

    spiral_rdpm_assignment_file = open(source + "/RDPmeans/spiral_rdpm_assignment.pkl", "rb")
    spiral_rdpm_assignment = pickle.load(spiral_rdpm_assignment_file)
    spiral_rdpm_assignment_file.close()

    moons_rdpm_assignment_file = open(source + "/RDPmeans/moons_rdpm_assignment.pkl", "rb")
    moons_rdpm_assignment = pickle.load(moons_rdpm_assignment_file)
    moons_rdpm_assignment_file.close()

    circles_rdpm_assignment_file = open(source + "/RDPmeans/circles_rdpm_assignment.pkl", "rb")
    circles_rdpm_assignment = pickle.load(circles_rdpm_assignment_file)
    circles_rdpm_assignment_file.close()

    ############################### Load LCVQE results ###############################
    iris_lcvqe_assignment_file = open(source + "/LCVQE/iris_lcvqe_assignment.pkl", "rb")
    iris_lcvqe_assignment = pickle.load(iris_lcvqe_assignment_file)
    iris_lcvqe_assignment_file.close()

    wine_lcvqe_assignment_file = open(source + "/LCVQE/wine_lcvqe_assignment.pkl", "rb")
    wine_lcvqe_assignment = pickle.load(wine_lcvqe_assignment_file)
    wine_lcvqe_assignment_file.close()

    breast_cancer_lcvqe_assignment_file = open(source + "/LCVQE/breast_cancer_lcvqe_assignment.pkl", "rb")
    breast_cancer_lcvqe_assignment = pickle.load(breast_cancer_lcvqe_assignment_file)
    breast_cancer_lcvqe_assignment_file.close()

    glass_lcvqe_assignment_file = open(source + "/LCVQE/glass_lcvqe_assignment.pkl", "rb")
    glass_lcvqe_assignment = pickle.load(glass_lcvqe_assignment_file)
    glass_lcvqe_assignment_file.close()

    digits_lcvqe_assignment_file = open(source + "/LCVQE/digits_lcvqe_assignment.pkl", "rb")
    digits_lcvqe_assignment = pickle.load(digits_lcvqe_assignment_file)
    digits_lcvqe_assignment_file.close()

    rand_lcvqe_assignment_file = open(source + "/LCVQE/rand_lcvqe_assignment.pkl", "rb")
    rand_lcvqe_assignment = pickle.load(rand_lcvqe_assignment_file)
    rand_lcvqe_assignment_file.close()

    spiral_lcvqe_assignment_file = open(source + "/LCVQE/spiral_lcvqe_assignment.pkl", "rb")
    spiral_lcvqe_assignment = pickle.load(spiral_lcvqe_assignment_file)
    spiral_lcvqe_assignment_file.close()

    moons_lcvqe_assignment_file = open(source + "/LCVQE/moons_lcvqe_assignment.pkl", "rb")
    moons_lcvqe_assignment = pickle.load(moons_lcvqe_assignment_file)
    moons_lcvqe_assignment_file.close()

    circles_lcvqe_assignment_file = open(source + "/LCVQE/circles_lcvqe_assignment.pkl", "rb")
    circles_lcvqe_assignment = pickle.load(circles_lcvqe_assignment_file)
    circles_lcvqe_assignment_file.close()

    ############################### Load true labels ###############################
    iris_labels_file = open(source + "/trueLabels/iris_labels.pkl", "rb")
    iris_labels = pickle.load(iris_labels_file)
    iris_labels_file.close()

    wine_labels_file = open(source + "/trueLabels/wine_labels.pkl", "rb")
    wine_labels = pickle.load(wine_labels_file)
    wine_labels_file.close()

    breast_cancer_labels_file = open(source + "/trueLabels/breast_cancer_labels.pkl", "rb")
    breast_cancer_labels = pickle.load(breast_cancer_labels_file)
    breast_cancer_labels_file.close()

    glass_labels_file = open(source + "/trueLabels/glass_labels.pkl", "rb")
    glass_labels = pickle.load(glass_labels_file)
    glass_labels_file.close()

    digits_labels_file = open(source + "/trueLabels/digits_labels.pkl", "rb")
    digits_labels = pickle.load(digits_labels_file)
    digits_labels_file.close()

    rand_labels_file = open(source + "/trueLabels/rand_labels.pkl", "rb")
    rand_labels = pickle.load(rand_labels_file)
    rand_labels_file.close()

    spiral_labels_file = open(source + "/trueLabels/spiral_labels.pkl", "rb")
    spiral_labels = pickle.load(spiral_labels_file)
    spiral_labels_file.close()

    moons_labels_file = open(source + "/trueLabels/moons_labels.pkl", "rb")
    moons_labels = pickle.load(moons_labels_file)
    moons_labels_file.close()

    circles_labels_file = open(source + "/trueLabels/circles_labels.pkl", "rb")
    circles_labels = pickle.load(circles_labels_file)
    circles_labels_file.close()

    ############################### Get scores for Kmeans ###############################
    # iris_km_rand_score = adjusted_rand_score(iris_labels, iris_km_assignment)
    # wine_km_rand_score = adjusted_rand_score(wine_labels, wine_km_assignment)
    # breast_cancer_km_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_km_assignment)
    # glass_km_rand_score = adjusted_rand_score(glass_labels, glass_km_assignment)
    # digits_km_rand_score = adjusted_rand_score(digits_labels, digits_km_assignment)
    # rand_km_rand_score = adjusted_rand_score(rand_labels, rand_km_assignment)
    # spiral_km_rand_score = adjusted_rand_score(spiral_labels, spiral_km_assignment)
    # moons_km_rand_score = adjusted_rand_score(moons_labels, moons_km_assignment)
    # circles_km_rand_score = adjusted_rand_score(circles_labels, circles_km_assignment)
    # print("########################## K-means ##########################")
    # print([iris_km_rand_score, wine_km_rand_score, breast_cancer_km_rand_score,
    #        glass_km_rand_score, digits_km_rand_score, rand_km_rand_score,
    #        spiral_km_rand_score, moons_km_rand_score, circles_km_rand_score])

    ############################### Get scores for COP-Kmeans ###############################
    # iris_ckm_rand_score = adjusted_rand_score(iris_labels, iris_ckm_assignment)
    # wine_ckm_rand_score = adjusted_rand_score(wine_labels, wine_ckm_assignment)
    # breast_cancer_ckm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_ckm_assignment)
    # glass_ckm_rand_score = adjusted_rand_score(glass_labels, glass_ckm_assignment)
    # digits_ckm_rand_score = adjusted_rand_score(digits_labels, digits_ckm_assignment)
    # rand_ckm_rand_score = adjusted_rand_score(rand_labels, rand_ckm_assignment)
    # spiral_ckm_rand_score = adjusted_rand_score(spiral_labels, spiral_ckm_assignment)
    # moons_ckm_rand_score = adjusted_rand_score(moons_labels, moons_ckm_assignment)
    # circles_ckm_rand_score = adjusted_rand_score(circles_labels, circles_ckm_assignment)
    # print("########################## COPK-means ##########################")
    # print([iris_ckm_rand_score, wine_ckm_rand_score, breast_cancer_ckm_rand_score,
    #        glass_ckm_rand_score, digits_ckm_rand_score, rand_ckm_rand_score,
    #        spiral_ckm_rand_score, moons_ckm_rand_score, circles_ckm_rand_score])


    ############################### Get scores for CECM ###############################
    iris_cecm_rand_score = adjusted_rand_score(iris_labels, iris_cecm_assignment)
    wine_cecm_rand_score = adjusted_rand_score(wine_labels, wine_cecm_assignment)
    breast_cancer_cecm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_cecm_assignment)
    glass_cecm_rand_score = 90 #adjusted_rand_score(glass_labels, glass_cecm_assignment)
    digits_cecm_rand_score = 90 #adjusted_rand_score(digits_labels, digits_cecm_assignment)
    rand_cecm_rand_score = adjusted_rand_score(rand_labels, rand_cecm_assignment)
    spiral_cecm_rand_score = adjusted_rand_score(spiral_labels, spiral_cecm_assignment)
    moons_cecm_rand_score = adjusted_rand_score(moons_labels, moons_cecm_assignment)
    circles_cecm_rand_score = adjusted_rand_score(circles_labels, circles_cecm_assignment)

    print("########################## CECM ##########################")
    print([iris_cecm_rand_score, wine_cecm_rand_score, breast_cancer_cecm_rand_score,
           glass_cecm_rand_score, digits_cecm_rand_score, rand_cecm_rand_score,
           spiral_cecm_rand_score, moons_cecm_rand_score, circles_cecm_rand_score])

    ############################### Get scores for TVclust ###############################
    iris_tvclust_rand_score = adjusted_rand_score(iris_labels, iris_tvclust_assignment)
    wine_tvclust_rand_score = adjusted_rand_score(wine_labels, wine_tvclust_assignment)
    breast_cancer_tvclust_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_tvclust_assignment)
    glass_tvclust_rand_score = adjusted_rand_score(glass_labels, glass_tvclust_assignment)
    digits_tvclust_rand_score = adjusted_rand_score(digits_labels, digits_tvclust_assignment)
    rand_tvclust_rand_score = adjusted_rand_score(rand_labels, rand_tvclust_assignment)
    spiral_tvclust_rand_score = adjusted_rand_score(spiral_labels, spiral_tvclust_assignment)
    moons_tvclust_rand_score = adjusted_rand_score(moons_labels, moons_tvclust_assignment)
    circles_tvclust_rand_score = adjusted_rand_score(circles_labels, circles_tvclust_assignment)

    print("########################## TV clust ##########################")
    print([iris_tvclust_rand_score, wine_tvclust_rand_score, breast_cancer_tvclust_rand_score,
           glass_tvclust_rand_score, digits_tvclust_rand_score, rand_tvclust_rand_score,
           spiral_tvclust_rand_score, moons_tvclust_rand_score, circles_tvclust_rand_score])

    ############################### Get scores for RDPM ###############################
    iris_rdpm_rand_score = adjusted_rand_score(iris_labels, iris_rdpm_assignment)
    wine_rdpm_rand_score = adjusted_rand_score(wine_labels, wine_rdpm_assignment)
    breast_cancer_rdpm_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_rdpm_assignment)
    glass_rdpm_rand_score = adjusted_rand_score(glass_labels, glass_rdpm_assignment)
    digits_rdpm_rand_score = adjusted_rand_score(digits_labels, digits_rdpm_assignment)
    rand_rdpm_rand_score = adjusted_rand_score(rand_labels, rand_rdpm_assignment)
    spiral_rdpm_rand_score = adjusted_rand_score(spiral_labels, spiral_rdpm_assignment)
    moons_rdpm_rand_score = adjusted_rand_score(moons_labels, moons_rdpm_assignment)
    circles_rdpm_rand_score = adjusted_rand_score(circles_labels, circles_rdpm_assignment)

    print("########################## RDPM ##########################")
    print([iris_rdpm_rand_score, wine_rdpm_rand_score, breast_cancer_rdpm_rand_score,
           glass_rdpm_rand_score, digits_rdpm_rand_score, rand_rdpm_rand_score,
           spiral_rdpm_rand_score, moons_rdpm_rand_score, circles_rdpm_rand_score])

    ############################### Get scores for LCVQE ###############################
    iris_lcvqe_rand_score = adjusted_rand_score(iris_labels, iris_lcvqe_assignment)
    wine_lcvqe_rand_score = adjusted_rand_score(wine_labels, wine_lcvqe_assignment)
    breast_cancer_lcvqe_rand_score = adjusted_rand_score(breast_cancer_labels, breast_cancer_lcvqe_assignment)
    glass_lcvqe_rand_score = adjusted_rand_score(glass_labels, glass_lcvqe_assignment)
    digits_lcvqe_rand_score = adjusted_rand_score(digits_labels, digits_lcvqe_assignment)
    rand_lcvqe_rand_score = adjusted_rand_score(rand_labels, rand_lcvqe_assignment)
    spiral_lcvqe_rand_score = adjusted_rand_score(spiral_labels, spiral_lcvqe_assignment)
    moons_lcvqe_rand_score = adjusted_rand_score(moons_labels, moons_lcvqe_assignment)
    circles_lcvqe_rand_score = adjusted_rand_score(circles_labels, circles_lcvqe_assignment)

    print("########################## LCVQE ##########################")
    print([iris_lcvqe_rand_score, wine_lcvqe_rand_score, breast_cancer_lcvqe_rand_score,
           glass_lcvqe_rand_score, digits_lcvqe_rand_score, rand_lcvqe_rand_score,
           spiral_lcvqe_rand_score, moons_lcvqe_rand_score, circles_lcvqe_rand_score])



if __name__ == "__main__": main()
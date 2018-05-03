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

def main():
    colors = ['b', 'orange', 'g', 'r', 'Brown', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    rand_set, rand_labels = generate_data_2D([[3, 1], [2, 2], [3, 3]], [[0.08, 0.08], [0.08, 0.08], [0.08, 0.08]], 100)

    rand_set2, rand_labels2 = generate_data_2D([[1.5, 1.5], [2.1, 2.1]], [[0.1, 0.1], [0.1, 0.1]], 100)
    rand_set2 = np.vstack((rand_set2, [1.7,1.7]))
    rand_set2 = np.vstack((rand_set2, [1.8, 1.8]))
    rand_set2 = np.vstack((rand_set2, [1.85, 1.85]))
    rand_set2 = np.vstack((rand_set2, [1.9, 1.9]))
    moons_set, moons_labels = datasets.make_moons(300, .5, .05, 45)
    moons_set += 1.5

    rand = np.random.rand(1000, 2)
    rand_km_assignment = KMeans(init="random", n_clusters=4).fit(rand).labels_

    # fig0, ax0 = plt.subplots()
    # ax0.plot(rand_set[: , 0], rand_set[:, 1], '.', color='black', markersize = 10)
    #
    # plt.axis('off')
    # plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # ax1.plot(moons_set[:, 0], moons_set[:, 1], '.', color='black', markersize=10)
    #
    # plt.axis('off')
    # plt.show()


    fig2, ax2 = plt.subplots()
    ax2.plot(rand[:, 0][rand_km_assignment == 0], rand[:, 1][rand_km_assignment == 0], '.', color=colors[0])
    ax2.plot(rand[:, 0][rand_km_assignment == 1], rand[:, 1][rand_km_assignment == 1], '.', color=colors[1])
    ax2.plot(rand[:, 0][rand_km_assignment == 2], rand[:, 1][rand_km_assignment == 2], '.', color=colors[2])
    ax2.plot(rand[:, 0][rand_km_assignment == 3], rand[:, 1][rand_km_assignment == 3], '.', color=colors[3])
    #ax2.plot(rand[:, 0], rand[:, 1], '.', color='black', markersize=10)
    plt.ylim(0,1.01)
    plt.xlim(0, 1.01)
    plt.axis('off')
    plt.show()

if __name__ == "__main__": main()
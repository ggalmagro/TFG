# for i in range(np.shape(iris_const)[0]):
    #     for j in range(np.shape(iris_const)[1]):
    #
    #         if iris_const[i, j] == 1 and iris_labels[i] != iris_labels[j]:
    #             print("Error en las restricciones")
    #
    #         if iris_const[i, j] == -1 and iris_labels[i] == iris_labels[j]:
    #             print("Error en las restricciones")
    #
    # ax0 = draw_data_2DNC(iris_set, iris_labels, 3, "title")
    # ax1 = draw_data_2DNC(iris_set, iris_labels, 3, "title")
    # c1, c2 = draw_const(iris_set, iris_const, ax0, ax1)
    # plt.show()
    #
    # ax0 = draw_data_2DNC(rand_set, rand_labels, 3, "title")
    # ax1 = draw_data_2DNC(rand_set, rand_labels, 3, "title")
    # c1, c2 = draw_const(rand_set, rand_const, ax0, ax1)
    # plt.show()
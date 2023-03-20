import math
import os
from matplotlib import pyplot as plt
import numpy as np


def plot_reinforcement(Y1, Y2, Y1_std, Y2_std):

    X1 = range(len(Y1))
    X2 = range(len(Y2))
    fig, ax = plt.subplots(1)

    # X1 = [X1[0], X1[10], X1[20], X1[30], X1[40], X1[50], X1[60], X1[70], X1[80], X1[90], X1[100], X1[110], X1[120]]
    # Y1 = [Y1[0], Y1[10], Y1[20], Y1[30], Y1[40], Y1[50], Y1[60], Y1[70], Y1[80], Y1[90], Y1[100], Y1[110], Y1[120]]
    # Y1_std = [Y1_std[0], Y1_std[10], Y1_std[20], Y1_std[30], Y1_std[40], Y1_std[50], Y1_std[60], Y1_std[70],
    #                    Y1_std[80], Y1_std[90], Y1_std[100], Y1_std[110], Y1_std[120]]
    # X2 = [X2[0], X2[10], X2[20], X2[30], X2[40], X2[50], X2[60], X2[70], X2[80], X2[90], X2[100], X2[110], X2[120]]
    # Y2 = [Y2[0], Y2[10], Y2[20], Y2[30], Y2[40], Y2[50], Y2[60], Y2[70], Y2[80], Y2[90], Y2[100], Y2[110], Y2[120]]
    # Y2_std = [Y2_std[0], Y2_std[10], Y2_std[20], Y2_std[30], Y2_std[40], Y2_std[50], Y2_std[60], Y2_std[70],
    #                    Y2_std[80], Y2_std[90], Y2_std[100], Y2_std[110], Y2_std[120]]

    ax.plot(X2, Y2, label= "Social Acceptance + Fairness" , marker="." , color='royalblue')
    ax.plot(X1, Y1, label= "Fairness", marker="_", color='red')

    ax.fill_between(np.array(X1), np.array(Y1) - np.array(Y1_std), np.array(Y1) + np.array(Y1_std), facecolor='red', alpha=0.5)
    ax.fill_between(np.array(X2), np.array(Y2) - np.array(Y2_std), np.array(Y2) + np.array(Y2_std), facecolor='royalblue', alpha=0.5)

    ax.grid()
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cumulative AVG Reward of Sampled LA Networks')

    plt.show()


def con_to_int(my_list):
    new_list = []
    for i in my_list:
        i.strip("\n")
        new_list.append(int(i))
    return new_list


dir_path = "learning_trace/"
loss1_penals = []
loss2_penals = []

for f in os.listdir(dir_path):
    with open(dir_path + "/" + f) as loss_f:
        if f.startswith("loss_1"):
            loss1_penals.append(con_to_int(loss_f.readlines()))

        if f.startswith("loss_2"):
            loss2_penals.append(con_to_int(loss_f.readlines()))


loss_1_Y = []
loss_2_Y = []
loss_1_Y_e = []
loss_2_Y_e = []

element_of_intrst = 0
exit_cond = max([len(l) for l in loss1_penals])

while True:
    element_value = []
    counter = 0
    for l in loss1_penals:
        if element_of_intrst < len(l):
            element_value.append((l[element_of_intrst]))
            counter += 1

    e = np.std(element_value)
    element_value = np.mean(element_value)
    loss_1_Y.append(element_value)
    loss_1_Y_e.append(e)
    if len(loss_1_Y) == exit_cond:
        break
    else:
        element_of_intrst += 1


element_of_intrst = 0
exit_cond = max([len(l) for l in loss2_penals])

while True:
    element_value = []
    counter = 0
    for l in loss2_penals:
        if element_of_intrst < len(l):
            element_value.append((l[element_of_intrst]))
            counter += 1

    e = np.std(element_value)
    element_value = np.mean(element_value)
    loss_2_Y.append(element_value)
    loss_2_Y_e.append(e)
    if len(loss_2_Y) == exit_cond:
        break
    else:
        element_of_intrst += 1

print(loss_1_Y_e)
print(loss_2_Y_e)
m = min(len(loss_1_Y), len(loss_2_Y), len(loss_1_Y_e), len(loss_2_Y_e))
plot_reinforcement(loss_1_Y[:m], loss_2_Y[:m], loss_1_Y_e[:m], loss_2_Y_e[:m])
#plot_reinforcement(loss_1_Y, loss_2_Y, loss_1_Y_e, loss_2_Y_e)

import math
import os
from matplotlib import pyplot as plt
import numpy as np


def plot_entropy(Y1, Y2):

    X1 = range(len(Y1))
    X2 = range(len(Y2))
    fig, ax = plt.subplots(1)


    ax.plot(X2, Y2, label= "Social Acceptance + Fairness" , marker="." , color='royalblue')
    ax.plot(X2, [np.mean(Y2) for _ in Y2],  color='royalblue')

    ax.plot(X1, Y1, label= "Fairness" , marker="_", color='red')
    ax.plot(X1, [np.mean(Y1) for _ in Y1], color='red')


    ax.grid()
    ax.legend()
    ax.set_xlabel('Indices of Users with Different Entropy')
    ax.set_ylabel("Campaign's Incentives Entropy")
    plt.show()


def con_to_int(myh_list):
    v = []
    for i in myh_list:
        v.append(float(i.split()[-1].strip("\n")))
    return v

dir_path = "entropy/"
Y1 = []
Y2= []
for f in os.listdir(dir_path):
    with open(dir_path + "/" + f) as _f:
        if f.endswith("_1.txt"):
            Y1 = con_to_int(_f.readlines())


        if f.endswith("_2.txt"):
            Y2 = con_to_int(_f.readlines())

print(Y1)
print(Y2)

final_Y1 = []
final_Y2 = []
for e, y in enumerate(Y1):
    if y != Y2[e]:
        final_Y1.append(y)
        final_Y2.append(Y2[e])

plot_entropy(final_Y1, final_Y2)
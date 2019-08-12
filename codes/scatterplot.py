################################################################################
# plots a scatter graph
# between 2 data series, ONLY 64 DIMENSIONS in this version
################################################################################

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


def scatterplot(data, x, y, output):

    df = pd.read_csv(data, index_col=0)

    #df8 = df[df["dimension"] == 8]
    #df16 = df[df["dimension"] == 16]
    #df32 = df[df["dimension"] == 32]
    df64 = df[df["dimension"] == 64]

    fig = plt.figure(figsize=(2.6,2.4))

    #ax1 = fig.add_subplot(221)
    #ax2 = fig.add_subplot(222)
    #ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(111)

    #ax1.scatter(df8[x], df8[y], marker=".", c="black")
    #ax2.scatter(df16[x], df16[y], marker=".", c="black")
    #ax3.scatter(df32[x], df32[y], marker=".", c="black")
    ax4.scatter(df64[x], df64[y], marker=".", c="black", s=10)

    ax4.set_ylim((0,1))
    ax4.set_xlim((0,1))

    ax4.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    #ax1.set_title("8 dimensions")
    #ax2.set_title("16 dimensions")
    #ax3.set_title("32 dimensions")
    #ax4.set_title("64 dimensions")

    lab_dict = {"mrr":"MRR", "mrank":"MR", "MLP_acc":"classifier accuracy", \
    "MLP_wF1":"classifier F1 score"}

    ax4.set_xlabel(lab_dict[x])
    ax4.set_ylabel(lab_dict[y])

    fig.tight_layout()
    fig.savefig(output)

################################################################################

# data:         path of file containing the data to be examined
# x:            x-axis data
# y:            y-axis data
# output:       path and .csv file name for output

dss = ["FULL", "N2N", "PROC", "PTREEv2", "IHDv2", "IHPv2", "MUSHv2"]

for ds in dss:

	data = "./norange/new_remain/" + ds + "_jointable_norange.csv"
	output = "./norange/new_remain/" + ds + "_scatterplot2_norange.png"

	scatterplot(data = data, \
    x = "MLP_wF1", \
    y = "mrr", \
    output = output)

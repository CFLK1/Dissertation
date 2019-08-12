################################################################################
# calculates Spearman rank-order correlation coefficient r
# between 2 data series
################################################################################

import numpy as np
import pandas as pd
import sys
import scipy.stats as stats


def spearman_r(data, output):

    df = pd.read_csv(data, index_col=0)

    df8 = df[df["dimension"] == 8]
    df16 = df[df["dimension"] == 16]
    df32 = df[df["dimension"] == 32]
    df64 = df[df["dimension"] == 64]

    results = []

    dflist = [df8, df16, df32, df64, df]

    for dat in dflist:

        r1 = stats.spearmanr(dat["mrr"], dat["MLP_acc"])[0]
        r2 = stats.spearmanr(dat["mrr"], dat["MLP_wF1"])[0]
        r3 = stats.spearmanr(dat["mrank"], dat["MLP_acc"])[0]
        r4 = stats.spearmanr(dat["mrank"], dat["MLP_wF1"])[0]
        r5 = stats.spearmanr(dat["inPGroupHitsAt1"], dat["MLP_acc"])[0]
        r6 = stats.spearmanr(dat["inPGroupHitsAt1"], dat["MLP_wF1"])[0]
        results.append([r1, r2, r3, r4, r5, r6])

    colnames = ["mrr_acc", "mrr_wF1", "mrank_acc", "mrank_wF1", "relH1_acc", \
                "relH1_wF1"]
    indices = ["dim8", "dim16", "dim32", "dim64", "alldim"]
    df_out = pd.DataFrame(results, columns=colnames, index=indices)

    df_out.to_csv(output)


################################################################################

# data:         path of file containing the data to be examined
# output:       path and .csv file name for output

spearman_r(data = "./norange/new_remain/FULL_jointable_norange.csv", \
          output = "./norange/new_remain/FULL_spearmanr_norange.csv")

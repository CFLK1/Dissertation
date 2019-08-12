################################################################################
# hyperparameter % in top 50/100
################################################################################
import numpy as np

import os
import sys

input = "./norange/new_remain/PROC_jointable_norange.csv"

df = pd.read_csv(input)

# edit if not all hp options exist (eg. B type, range)


out_colnames = ["alldim", "dim8", "dim16", "dim32", "dim64"]

out_100 = "./norange/new_remain/PROC_top100_acc_norange.csv"




#############################################################




    HType = df.HType.value_counts().sort_index()/len(df.index)
    ABType = df.ABType.value_counts().sort_index()/len(df.index)
    dimension = df.dimension.value_counts().sort_index()/len(df.index)
    learnFac = df.learnFac.value_counts().sort_index()/len(df.index)
    margin = df.margin.value_counts().sort_index()/len(df.index)
    constr = df.constr.value_counts().sort_index()/len(df.index)
    LType = df.LType.value_counts().sort_index()/len(df.index)

    # assemble
    hp_series = pd.concat([HType, ABType, dimension, learnFac, margin, constr, LType])
    hp = pd.DataFrame(hp_series)
    return hp




################################################################################
########################## 8 dim ###########################
############################################################




hp_100_d8 = topN_hp(by_metric_100_d8)

########################## 16 dim ###########################
#############################################################
df_d16 = df[df["dimension"] == 16]






#############################################################






########################## 64 dim ###########################
#############################################################





################################################################################
########################## assemble ###########################
###############################################################















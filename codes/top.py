################################################################################

# hyperparameter % in top 50/100

################################################################################



import numpy as np


import pandas as pd

import os

import sys




input = "./norange/new_remain/PROC_jointable_norange.csv"




df = pd.read_csv(input)




# edit if not all hp options exist (eg. B type, range)

out_index = ["H1","H2","H3","H4","A","B",8,16,32,64,10,100,


             1000,1.0,2.0,4.0,"fixed","max","L0","L1","L2"]

out_colnames = ["alldim", "dim8", "dim16", "dim32", "dim64"]




out_50 = "./norange/new_remain/PROC_top50_acc_norange.csv"

out_100 = "./norange/new_remain/PROC_top100_acc_norange.csv"





metric = "MLP_acc"



################################################################################


########################## alldim ###########################

###################
#############################################################




by_metric_50 = df.sort_values(metric, ascending=False).iloc[:50,:9].reset_index(drop=True)


by_metric_100 = df.sort_values(metric, ascending=False).iloc[:100,:9].reset_index(drop=True)





def topN_hp(df):
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





hp_50_alldim = topN_hp(by_metric_50)


hp_100_alldim = topN_hp(by_metric_100)




################################################################################

########################## 8 dim ###########################

####################
############################################################




df_d8 = df[df["dimension"] == 8]


by_metric_50_d8 = df_d8.sort_values(metric, ascending=False).iloc[:50,:9].reset_index(drop=True)


by_metric_100_d8 = df_d8.sort_values(metric, ascending=False).iloc[:100,:9].reset_index(drop=True)




hp_50_d8 = topN_hp(by_metric_50_d8)

hp_100_d8 = topN_hp(by_metric_100_d8)




################################################################################

########################## 16 dim ###########################

###################
#############################################################



df_d16 = df[df["dimension"] == 16]


by_metric_50_d16 = df_d16.sort_values(metric, ascending=False).iloc[:50,:9].reset_index(drop=True)


by_metric_100_d16 = df_d16.sort_values(metric, ascending=False).iloc[:100,:9].reset_index(drop=True)




hp_50_d16 = topN_hp(by_metric_50_d16)

hp_100_d16 = topN_hp(by_metric_100_d16)




################################################################################


########################## 32 dim ###########################

###################
#############################################################




df_d32 = df[df["dimension"] == 32]


by_metric_50_d32 = df_d32.sort_values(metric, ascending=False).iloc[:50,:9].reset_index(drop=True)


by_metric_100_d32 = df_d32.sort_values(metric, ascending=False).iloc[:100,:9].reset_index(drop=True)



hp_50_d32 = topN_hp(by_metric_50_d32)

hp_100_d32 = topN_hp(by_metric_100_d32)




################################################################################

########################## 64 dim ###########################

###################
#############################################################




df_d64 = df[df["dimension"] == 64]


by_metric_50_d64 = df_d64.sort_values(metric, ascending=False).iloc[:50,:9].reset_index(drop=True)


by_metric_100_d64 = df_d64.sort_values(metric, ascending=False).iloc[:100,:9].reset_index(drop=True)




hp_50_d64 = topN_hp(by_metric_50_d64)


hp_100_d64 = topN_hp(by_metric_100_d64)



################################################################################

########################## assemble ###########################

#################
###############################################################



hp_50 = pd.merge(hp_50_alldim, hp_50_d8, left_index=True, right_index=True, how="outer")


hp_50 = hp_50.merge(hp_50_d16, left_index=True, right_index=True, how="outer")


hp_50 = hp_50.merge(hp_50_d32, left_index=True, right_index=True, how="outer")


hp_50 = hp_50.merge(hp_50_d64, left_index=True, right_index=True, how="outer")


hp_50 = hp_50.reindex(out_index, axis=0).fillna(0)


hp_50.columns = out_colnames




hp_50.to_csv(out_50)





hp_100 = pd.merge(hp_100_alldim, hp_100_d8, left_index=True, right_index=True, how="outer")


hp_100 = hp_100.merge(hp_100_d16, left_index=True, right_index=True, how="outer")


hp_100 = hp_100.merge(hp_100_d32, left_index=True, right_index=True, how="outer")


hp_100 = hp_100.merge(hp_100_d64, left_index=True, right_index=True, how="outer")


hp_100 = hp_100.reindex(out_index, axis=0).fillna(0)


hp_100.columns = out_colnames




hp_100.to_csv(out_100)


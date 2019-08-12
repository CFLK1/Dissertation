################################################################################
# combines MLP results with e-models scores
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from pandasql import PandaSQL


def join_emodel_MLP(emodel, mlp, output, special_col=False):

    df_emodel = pd.read_csv(emodel, index_col=0)
    df_mlp = pd.read_csv(mlp)

    pdsql = PandaSQL()

################################################################################
################################## SQL Query ###################################
################################################################################

    if special_col == False:

        join_mlp = """
            SELECT a.HType, a.ABType, a.dimension, a.learnFac, a.margin, a.constr,
                a.LType, a.hitsAt1, a.hitsAt2, a.hitsAt3, a.hitsAt5, a.hitsAtX, a.mrank,
                a.mrr, a.off1, a.off95, a.off9, b.MLP_acc, b.MLP_wF1
            FROM df_emodel a
            LEFT OUTER JOIN df_mlp b ON
                a.HType = b.HType
                AND a.ABType = b.ABType
                AND a.dimension = b.dimension
                AND a.learnFac = b.learnFac
                AND a.margin = b.margin
                AND a.constr = b.constr
                AND a.LType = b.LType
            ;
            """
        jointable = pdsql(join_mlp, locals())

        jointable.columns = ["HType", "ABType", "dimension", "learnFac", "margin", \
        "constr", "LType", "hitsAt1", "hitsAt2", "hitsAt3", "hitsAt5", "hitsAtX", \
        "mrank", "mrr", "off1", "off95", "off9", "MLP_acc", "MLP_wF1"]

        jointable.to_csv(output)

    else:

        df_special = pd.read_csv(special_col, index_col=0)
        df_emodel["targetHitsAt1"] = df_special.iloc[:, 0]

        join_mlp = """
            SELECT a.HType, a.ABType, a.dimension, a.learnFac, a.margin, a.constr,
                a.LType, a.hitsAt1, a.hitsAt2, a.hitsAt3, a.hitsAt5, a.hitsAtX, a.mrank,
                a.mrr, a.off1, a.off95, a.off9, a.targetHitsAt1, b.MLP_acc, b.MLP_wF1
            FROM df_emodel a
            LEFT OUTER JOIN df_mlp b ON
                a.HType = b.HType
                AND a.ABType = b.ABType
                AND a.dimension = b.dimension
                AND a.learnFac = b.learnFac
                AND a.margin = b.margin
                AND a.constr = b.constr
                AND a.LType = b.LType
            ;
            """
        jointable = pdsql(join_mlp, locals())

        jointable.columns = ["HType", "ABType", "dimension", "learnFac", "margin", \
        "constr", "LType", "hitsAt1", "hitsAt2", "hitsAt3", "hitsAt5", "hitsAtX", \
        "mrank", "mrr", "off1", "off95", "off9", "targetHitsAt1", "MLP_acc", "MLP_wF1"]

        jointable.to_csv(output)


################################################################################

# emodel:       path of file containing extracted emodel settings and scores
# mlp:          path of file containing MLP results
# output:       path and .csv file name for output
# special_col:  (optional) path of file containing a specific score to be included
#               as a column, default 'False'

join_emodel_MLP(emodel = "./norange/IHDv2_extracted_norange_withPaths.csv", \
                mlp = "./norange/new_remain/IHDv2_remaining_MLPacc_norange.csv", \
                output = "./norange/new_remain/IHDv2_jointable_norange.csv",
                special_col = "./norange/IHDv2_inSocialGroup_hitsAt5_norange.csv")

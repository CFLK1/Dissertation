################################################################################
# Performs statistical tests between metric-specific best hyperparameter options
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from pandasql import PandaSQL

input = "./norange/MUSHv2_extracted_norange.csv" # put extracted data file
# remember PTREEv2 has exceptions below !!!
output = "./norange/MUSHv2_stats_norange.csv" # define output file name

df = pd.read_csv(input)

pdsql = PandaSQL()

################################################################################
################################## q1: HType ###################################
################################################################################

q1_mrr = """
    SELECT b.HType, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxMRR = b.mrr
    ;
    """
q1_h1 = """
    SELECT b.HType, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q1_h3 = """
    SELECT b.HType, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q1_h5 = """
    SELECT b.HType, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q1_h10 = """
    SELECT b.HType, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q1_mr = """
    SELECT b.HType, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinMRank = b.mrank
    ;
    """

HType_MRR = pdsql(q1_mrr, locals())
HType_H1 = pdsql(q1_h1, locals())
HType_H3 = pdsql(q1_h3, locals())
HType_H5 = pdsql(q1_h5, locals())
HType_H10 = pdsql(q1_h10, locals())
HType_MRank = pdsql(q1_mr, locals())

#fig = plt.figure(figsize=(10, 10))
#fig.suptitle("Score distribution of HTypes by MRR")
#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
#ax3 = fig.add_subplot(223)
#ax4 = fig.add_subplot(224)
#ax1.hist(HType_MRR[HType_MRR["HType"] == "H1"].MaxMRR)
#ax2.hist(HType_MRR[HType_MRR["HType"] == "H2"].MaxMRR)
#ax3.hist(HType_MRR[HType_MRR["HType"] == "H3"].MaxMRR)
#ax4.hist(HType_MRR[HType_MRR["HType"] == "H4"].MaxMRR)
#ax1.set_title("H1")
#ax2.set_title("H2")
#ax3.set_title("H3")
#ax4.set_title("H4")
#plt.show()

list_of_df = [HType_MRR, HType_H1, HType_H3, HType_H5, HType_H10, HType_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

for i, df2 in enumerate(list_of_df):
    H1 = df2[df2.iloc[:,0] == "H1"].iloc[:,1].values
    H2 = df2[df2.iloc[:,0] == "H2"].iloc[:,1].values
    H3 = df2[df2.iloc[:,0] == "H3"].iloc[:,1].values
    H4 = df2[df2.iloc[:,0] == "H4"].iloc[:,1].values

    p_value_1 = round(stats.mannwhitneyu(H1, H2)[1], 4)
    p_value_2 = round(stats.mannwhitneyu(H1, H3)[1], 4)
    p_value_3 = round(stats.mannwhitneyu(H1, H4)[1], 4)
    p_value_4 = round(stats.mannwhitneyu(H2, H3)[1], 4)
    p_value_5 = round(stats.mannwhitneyu(H2, H4)[1], 4)
    p_value_6 = round(stats.mannwhitneyu(H3, H4)[1], 4)

    HType_part_stats_list = [(p_value_1, p_value_2, p_value_3),(np.nan, p_value_4, p_value_5), (np.nan, np.nan, p_value_6)]
    HType_part_stats = pd.DataFrame(HType_part_stats_list, index = ["H1", "H2", "H3"], columns = ["H2", "H3", "H4"])
    if i == 0:
        HType_stats = HType_part_stats
    else:
        HType_stats = HType_stats.merge(HType_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    HType_stats.insert(12, "x", [np.nan, np.nan, np.nan])
    HType_stats.insert(12, "y", [np.nan, np.nan, np.nan])
    HType_stats.insert(12, "z", [np.nan, np.nan, np.nan])

HType_stats.columns = ["H2MRR", "H3MRR", "H4MRR", "H2Hits1", "H3Hits1", "H4Hits1", "H2Hits3", "H3Hits3", "H4Hits3", "H2Hits5", "H3Hits5", "H4Hits5", "H2Hits10", "H3Hits10", "H4Hits10", "H2MR", "H3MR", "H4MR"]

################################################################################
################################## q2: ABType ##################################
################################################################################

q2_mrr = """
    SELECT b.ABType, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxMRR = b.mrr
    ;
    """
q2_h1 = """
    SELECT b.ABType, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q2_h3 = """
    SELECT b.ABType, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q2_h5 = """
    SELECT b.ABType, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q2_h10 = """
    SELECT b.ABType, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q2_mr = """
    SELECT b.ABType, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinMRank = b.mrank
    ;
    """

ABType_MRR = pdsql(q2_mrr, locals())
ABType_H1 = pdsql(q2_h1, locals())
ABType_H3 = pdsql(q2_h3, locals())
ABType_H5 = pdsql(q2_h5, locals())
ABType_H10 = pdsql(q2_h10, locals())
ABType_MRank = pdsql(q2_mr, locals())

list_of_df = [ABType_MRR, ABType_H1, ABType_H3, ABType_H5, ABType_H10, ABType_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

if ABType_MRR.ABType.nunique() == 1: # handle exception when some options are missing

    ABType_stats = pd.DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)], index=["A"])

else:

    for i, df2 in enumerate(list_of_df):
        A = df2[df2.iloc[:,0] == "A"].iloc[:,1].values
        B = df2[df2.iloc[:,0] == "B"].iloc[:,1].values

        p_value_1 = round(stats.mannwhitneyu(A, B)[1], 4)

        ABType_part_stats = pd.DataFrame(p_value_1, index = ["A"], columns = ["B"])
        if i == 0:
            ABType_stats = ABType_part_stats
        else:
            ABType_stats = ABType_stats.merge(ABType_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    ABType_stats.insert(4, "x", [np.nan])

ABType_stats.columns = ["BMRR", "BHits1", "BHits3", "BHits5", "BHits10", "BMR"]

################################################################################
################################ q3: dimension #################################
################################################################################

q3_mrr = """
    SELECT b.dimension, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxMRR = b.mrr
    ;
    """
q3_h1 = """
    SELECT b.dimension, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q3_h3 = """
    SELECT b.dimension, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q3_h5 = """
    SELECT b.dimension, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q3_h10 = """
    SELECT b.dimension, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q3_mr = """
    SELECT b.dimension, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinMRank = b.mrank
    ;
    """

dimension_MRR = pdsql(q3_mrr, locals())
dimension_H1 = pdsql(q3_h1, locals())
dimension_H3 = pdsql(q3_h3, locals())
dimension_H5 = pdsql(q3_h5, locals())
dimension_H10 = pdsql(q3_h10, locals())
dimension_MRank = pdsql(q3_mr, locals())

list_of_df = [dimension_MRR, dimension_H1, dimension_H3, dimension_H5, dimension_H10, dimension_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

for i, df2 in enumerate(list_of_df):
    D8 = df2[df2.iloc[:,0] == 8].iloc[:,1].values
    D16 = df2[df2.iloc[:,0] == 16].iloc[:,1].values
    D32 = df2[df2.iloc[:,0] == 32].iloc[:,1].values
    D64 = df2[df2.iloc[:,0] == 64].iloc[:,1].values

    p_value_1 = round(stats.mannwhitneyu(D8, D16)[1], 4)
    p_value_2 = round(stats.mannwhitneyu(D8, D32)[1], 4)
    p_value_3 = round(stats.mannwhitneyu(D8, D64)[1], 4)
    p_value_4 = round(stats.mannwhitneyu(D16, D32)[1], 4)
    p_value_5 = round(stats.mannwhitneyu(D16, D64)[1], 4)
    p_value_6 = round(stats.mannwhitneyu(D32, D64)[1], 4)

    dimension_part_stats_list = [(p_value_1, p_value_2, p_value_3),(np.nan, p_value_4, p_value_5), (np.nan, np.nan, p_value_6)]
    dimension_part_stats = pd.DataFrame(dimension_part_stats_list, index = [8, 16, 32], columns = [16, 32, 64])
    if i == 0:
        dimension_stats = dimension_part_stats
    else:
        dimension_stats = dimension_stats.merge(dimension_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    dimension_stats.insert(12, "x", [np.nan, np.nan, np.nan])
    dimension_stats.insert(12, "y", [np.nan, np.nan, np.nan])
    dimension_stats.insert(12, "z", [np.nan, np.nan, np.nan])

dimension_stats.columns = ["16MRR", "32MRR", "64MRR", "16Hits1", "32Hits1", "64Hits1", "16Hits3", "32Hits3", "64Hits3", "16Hits5", "32Hits5", "64Hits5", "16Hits10", "32Hits10", "64Hits10", "16MR", "32MR", "64MR"]

################################################################################
################################## q4: LType ###################################
################################################################################

q4_mrr = """
    SELECT b.LType, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxMRR = b.mrr
    ;
    """
q4_h1 = """
    SELECT b.LType, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q4_h3 = """
    SELECT b.LType, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q4_h5 = """
    SELECT b.LType, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q4_h10 = """
    SELECT b.LType, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q4_mr = """
    SELECT b.LType, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MinMRank = b.mrank
    ;
    """

LType_MRR = pdsql(q4_mrr, locals())
LType_H1 = pdsql(q4_h1, locals())
LType_H3 = pdsql(q4_h3, locals())
LType_H5 = pdsql(q4_h5, locals())
LType_H10 = pdsql(q4_h10, locals())
LType_MRank = pdsql(q4_mr, locals())

list_of_df = [LType_MRR, LType_H1, LType_H3, LType_H5, LType_H10, LType_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

for i, df2 in enumerate(list_of_df):
    L0 = df2[df2.iloc[:,0] == "L0"].iloc[:,1].values
    L1 = df2[df2.iloc[:,0] == "L1"].iloc[:,1].values
    L2 = df2[df2.iloc[:,0] == "L2"].iloc[:,1].values

    p_value_1 = round(stats.mannwhitneyu(L0, L1)[1], 4)
    p_value_2 = round(stats.mannwhitneyu(L0, L2)[1], 4)
    p_value_3 = round(stats.mannwhitneyu(L1, L2)[1], 4)

    LType_part_stats_list = [(p_value_1, p_value_2),(np.nan, p_value_3)]
    LType_part_stats = pd.DataFrame(LType_part_stats_list, index = ["L0", "L1"], columns = ["L1", "L2"])
    if i == 0:
        LType_stats = LType_part_stats
    else:
        LType_stats = LType_stats.merge(LType_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    LType_stats.insert(8, "x", [np.nan, np.nan])
    LType_stats.insert(8, "y", [np.nan, np.nan])

LType_stats.columns = ["L1MRR", "L2MRR", "L1Hits1", "L2Hits1", "L1Hits3", "L2Hits3", "L1Hits5", "L2Hits5", "L1Hits10", "L2Hits10", "L1MR", "L2MR"]

################################################################################
################################# q5: learnFac #################################
################################################################################

q5_mrr = """
    SELECT b.learnFac, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxMRR = b.mrr
    ;
    """
q5_h1 = """
    SELECT b.learnFac, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q5_h3 = """
    SELECT b.learnFac, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q5_h5 = """
    SELECT b.learnFac, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q5_h10 = """
    SELECT b.learnFac, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q5_mr = """
    SELECT b.learnFac, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinMRank = b.mrank
    ;
    """

learnFac_MRR = pdsql(q5_mrr, locals())
learnFac_H1 = pdsql(q5_h1, locals())
learnFac_H3 = pdsql(q5_h3, locals())
learnFac_H5 = pdsql(q5_h5, locals())
learnFac_H10 = pdsql(q5_h10, locals())
learnFac_MRank = pdsql(q5_mr, locals())

list_of_df = [learnFac_MRR, learnFac_H1, learnFac_H3, learnFac_H5, learnFac_H10, learnFac_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

for i, df2 in enumerate(list_of_df):
    LF10 = df2[df2.iloc[:,0] == 10].iloc[:,1].values
    LF100 = df2[df2.iloc[:,0] == 100].iloc[:,1].values
    LF1000 = df2[df2.iloc[:,0] == 1000].iloc[:,1].values

    p_value_1 = round(stats.mannwhitneyu(LF10, LF100)[1], 4)
    p_value_2 = round(stats.mannwhitneyu(LF10, LF1000)[1], 4)
    p_value_3 = round(stats.mannwhitneyu(LF100, LF1000)[1], 4)

    learnFac_part_stats_list = [(p_value_1, p_value_2),(np.nan, p_value_3)]
    learnFac_part_stats = pd.DataFrame(learnFac_part_stats_list, index = [10, 100], columns = [100, 1000])
    if i == 0:
        learnFac_stats = learnFac_part_stats
    else:
        learnFac_stats = learnFac_stats.merge(learnFac_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    learnFac_stats.insert(8, "x", [np.nan, np.nan])
    learnFac_stats.insert(8, "y", [np.nan, np.nan])

learnFac_stats.columns = ["100MRR", "1000MRR", "100Hits1", "1000Hits1", "100Hits3", "1000Hits3", "100Hits5", "1000Hits5", "100Hits10", "1000Hits10", "100MR", "1000MR"]

################################################################################
################################## q6: margin ##################################
################################################################################

q6_mrr = """
    SELECT b.margin, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxMRR = b.mrr
    ;
    """
q6_h1 = """
    SELECT b.margin, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q6_h3 = """
    SELECT b.margin, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q6_h5 = """
    SELECT b.margin, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q6_h10 = """
    SELECT b.margin, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q6_mr = """
    SELECT b.margin, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MinMRank = b.mrank
    ;
    """

margin_MRR = pdsql(q6_mrr, locals())
margin_H1 = pdsql(q6_h1, locals())
margin_H3 = pdsql(q6_h3, locals())
margin_H5 = pdsql(q6_h5, locals())
margin_H10 = pdsql(q6_h10, locals())
margin_MRank = pdsql(q6_mr, locals())

list_of_df = [margin_MRR, margin_H1, margin_H3, margin_H5, margin_H10, margin_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

for i, df2 in enumerate(list_of_df):
    M1 = df2[df2.iloc[:,0] == 1].iloc[:,1].values
    M2 = df2[df2.iloc[:,0] == 2].iloc[:,1].values
    M4 = df2[df2.iloc[:,0] == 4].iloc[:,1].values

    p_value_1 = round(stats.mannwhitneyu(M1, M2)[1], 4)
    p_value_2 = round(stats.mannwhitneyu(M1, M4)[1], 4)
    p_value_3 = round(stats.mannwhitneyu(M2, M4)[1], 4)

    margin_part_stats_list = [(p_value_1, p_value_2),(np.nan, p_value_3)]
    margin_part_stats = pd.DataFrame(margin_part_stats_list, index = [1, 2], columns = [2, 4])
    if i == 0:
        margin_stats = margin_part_stats
    else:
        margin_stats = margin_stats.merge(margin_part_stats, left_index=True, right_index=True)

if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    margin_stats.insert(8, "x", [np.nan, np.nan])
    margin_stats.insert(8, "y", [np.nan, np.nan])

margin_stats.columns = ["2MRR", "4MRR", "2Hits1", "4Hits1", "2Hits3", "4Hits3", "2Hits5", "4Hits5", "2Hits10", "4Hits10", "2MR", "4MR"]

################################################################################
################################## q7: constr ##################################
################################################################################

q7_mrr = """
    SELECT b.constr, MaxMRR
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MAX(mrr) AS MaxMRR
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MaxMRR = b.mrr
    ;
    """
q7_h1 = """
    SELECT b.constr, MaxHITS1
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MAX(hitsAt1) AS MaxHITS1
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MaxHITS1 = b.hitsAt1
    ;
    """
q7_h3 = """
    SELECT b.constr, MaxHITS3
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MAX(hitsAt3) AS MaxHITS3
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MaxHITS3 = b.hitsAt3
    ;
    """
q7_h5 = """
    SELECT b.constr, MaxHITS5
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MAX(hitsAt5) AS MaxHITS5
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MaxHITS5 = b.hitsAt5
    ;
    """
q7_h10 = """
    SELECT b.constr, MaxHITSX
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MAX(hitsAtX) AS MaxHITSX
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MaxHITSX = b.hitsAtX
    ;
    """
q7_mr = """
    SELECT b.constr, MinMRank
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MIN(mrank) AS MinMRank
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MinMRank = b.mrank
    ;
    """

constr_MRR = pdsql(q7_mrr, locals())
constr_H1 = pdsql(q7_h1, locals())
constr_H3 = pdsql(q7_h3, locals())
constr_H5 = pdsql(q7_h5, locals())
constr_H10 = pdsql(q7_h10, locals())
constr_MRank = pdsql(q7_mr, locals())

list_of_df = [constr_MRR, constr_H1, constr_H3, constr_H5, constr_H10, constr_MRank]
if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
    del list_of_df[-2]

if constr_MRR.constr.nunique() == 2: # handle exception when some options are missing
    for i, df2 in enumerate(list_of_df):
        fixed = df2[df2.iloc[:,0] == "fixed"].iloc[:,1].values
        max = df2[df2.iloc[:,0] == "max"].iloc[:,1].values

        p_value_1 = round(stats.mannwhitneyu(fixed, max)[1], 4)

        constr_part_stats = pd.DataFrame(p_value_1, index = ["fixed"], columns = ["max"])
        if i == 0:
            constr_stats = constr_part_stats
        else:
            constr_stats = constr_stats.merge(constr_part_stats, left_index=True, right_index=True)

    if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
        constr_stats.insert(4, "x", [np.nan])

    constr_stats.columns = ["maxMRR", "maxHits1", "maxHits3", "maxHits5", "maxHits10", "maxMR"]

else:
    for i, df2 in enumerate(list_of_df):
        fixed = df2[df2.iloc[:,0] == "fixed"].iloc[:,1].values
        max = df2[df2.iloc[:,0] == "max"].iloc[:,1].values
        range = df2[df2.iloc[:,0] == "range"].iloc[:,1].values

        p_value_1 = round(stats.mannwhitneyu(fixed, max)[1], 4)
        p_value_2 = round(stats.mannwhitneyu(fixed, range)[1], 4)
        p_value_3 = round(stats.mannwhitneyu(max, range)[1], 4)

        constr_part_stats_list = [(p_value_1, p_value_2),(np.nan, p_value_3)]
        constr_part_stats = pd.DataFrame(constr_part_stats_list, index = ["fixed", "max"], columns = ["max", "range"])
        if i == 0:
            constr_stats = constr_part_stats
        else:
            constr_stats = constr_stats.merge(constr_part_stats, left_index=True, right_index=True)

    if input == "./norange/PTREEv2_extracted_norange.csv": # handle exception when all scores are 1 (PTREE Hits10)
        constr_stats.insert(8, "x", [np.nan, np.nan])
        constr_stats.insert(8, "y", [np.nan, np.nan])

    constr_stats.columns = ["maxMRR", "rangeMRR", "maxHits1", "rangeHits1", "maxHits3", "rangeHits3", "maxHits5", "rangeHits5", "maxHits10", "rangeHits10", "maxMR", "rangeMR"]

################################################################################
################################ DATA ASSEMBLY #################################
################################################################################

tables = [HType_stats, ABType_stats, dimension_stats, LType_stats, learnFac_stats, margin_stats, constr_stats]
with open(output, "w") as csv_stream:
    for table in tables:
        table.to_csv(csv_stream, index=True)
        csv_stream.write("\n")

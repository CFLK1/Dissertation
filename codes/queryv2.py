################################################################################
# Performs query for analysis
################################################################################

import numpy as np
import pandas as pd
import sys

from pandasql import PandaSQL

input = "./norange/MUSHv2_extracted_norange.csv" # put extracted data file
output = "./norange/MUSHv2_hyperparameters_norange.csv" # define output file name

df = pd.read_csv(input)

pdsql = PandaSQL()

################################################################################
################################## q1: HType ###################################
################################################################################

q1_cyc = """
    SELECT b.HType, COUNT(b.HType) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    ABType, dimension, learnFac, margin, constr, LType,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY ABType, dimension, learnFac, margin, constr, LType) a ON
        a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.HType
    ;
    """
q1_mrr = """
    SELECT b.HType, COUNT(b.HType) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.HType
    ;
    """
q1_h1 = """
    SELECT b.HType, COUNT(b.HType) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.HType
    ;
    """
q1_h3 = """
    SELECT b.HType, COUNT(b.HType) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.HType
    ;
    """
q1_h5 = """
    SELECT b.HType, COUNT(b.HType) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.HType
    ;
    """
q1_h10 = """
    SELECT b.HType, COUNT(b.HType) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.HType
    ;
    """
q1_mr = """
    SELECT b.HType, COUNT(b.HType) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.HType
    ;
    """

HType_cycle = pdsql(q1_cyc, locals())
HType_MRR = pdsql(q1_mrr, locals())
HType_H1 = pdsql(q1_h1, locals())
HType_H3 = pdsql(q1_h3, locals())
HType_H5 = pdsql(q1_h5, locals())
HType_H10 = pdsql(q1_h10, locals())
HType_MRank = pdsql(q1_mr, locals())

q1_cyc_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY HType"""
q1_mrr_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY HType"""
q1_h1_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY HType"""
q1_h3_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY HType"""
q1_h5_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY HType"""
q1_h10_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY HType"""
q1_mr_all = """SELECT HType, COUNT(HType) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY HType"""

HType_cycle_all = pdsql(q1_cyc_all, locals())
HType_MRR_all = pdsql(q1_mrr_all, locals())
HType_H1_all = pdsql(q1_h1_all, locals())
HType_H3_all = pdsql(q1_h3_all, locals())
HType_H5_all = pdsql(q1_h5_all, locals())
HType_H10_all = pdsql(q1_h10_all, locals())
HType_MRank_all = pdsql(q1_mr_all, locals())

################################################################################
################################## q2: ABType ##################################
################################################################################

q2_cyc = """
    SELECT b.ABType, COUNT(b.ABType) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, dimension, learnFac, margin, constr, LType,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, dimension, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.dimension = b.dimension
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.ABType
    ;
    """
q2_mrr = """
    SELECT b.ABType, COUNT(b.ABType) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.ABType
    ;
    """
q2_h1 = """
    SELECT b.ABType, COUNT(b.ABType) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.ABType
    ;
    """
q2_h3 = """
    SELECT b.ABType, COUNT(b.ABType) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.ABType
    ;
    """
q2_h5 = """
    SELECT b.ABType, COUNT(b.ABType) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.ABType
    ;
    """
q2_h10 = """
    SELECT b.ABType, COUNT(b.ABType) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.ABType
    ;
    """
q2_mr = """
    SELECT b.ABType, COUNT(b.ABType) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.ABType
    ;
    """

ABType_cycle = pdsql(q2_cyc, locals())
ABType_MRR = pdsql(q2_mrr, locals())
ABType_H1 = pdsql(q2_h1, locals())
ABType_H3 = pdsql(q2_h3, locals())
ABType_H5 = pdsql(q2_h5, locals())
ABType_H10 = pdsql(q2_h10, locals())
ABType_MRank = pdsql(q2_mr, locals())

q2_cyc_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY ABType"""
q2_mrr_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY ABType"""
q2_h1_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY ABType"""
q2_h3_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY ABType"""
q2_h5_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY ABType"""
q2_h10_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY ABType"""
q2_mr_all = """SELECT ABType, COUNT(ABType) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY ABType"""

ABType_cycle_all = pdsql(q2_cyc_all, locals())
ABType_MRR_all = pdsql(q2_mrr_all, locals())
ABType_H1_all = pdsql(q2_h1_all, locals())
ABType_H3_all = pdsql(q2_h3_all, locals())
ABType_H5_all = pdsql(q2_h5_all, locals())
ABType_H10_all = pdsql(q2_h10_all, locals())
ABType_MRank_all = pdsql(q2_mr_all, locals())

################################################################################
################################ q3: dimension #################################
################################################################################

q3_cyc = """
    SELECT b.dimension, COUNT(b.dimension) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, learnFac, margin, constr, LType,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, ABType, learnFac, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.learnFac = b.learnFac
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.dimension
    ;
    """
q3_mrr = """
    SELECT b.dimension, COUNT(b.dimension) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.dimension
    ;
    """
q3_h1 = """
    SELECT b.dimension, COUNT(b.dimension) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.dimension
    ;
    """
q3_h3 = """
    SELECT b.dimension, COUNT(b.dimension) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.dimension
    ;
    """
q3_h5 = """
    SELECT b.dimension, COUNT(b.dimension) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.dimension
    ;
    """
q3_h10 = """
    SELECT b.dimension, COUNT(b.dimension) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.dimension
    ;
    """
q3_mr = """
    SELECT b.dimension, COUNT(b.dimension) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.dimension
    ;
    """

dimension_cycle = pdsql(q3_cyc, locals())
dimension_MRR = pdsql(q3_mrr, locals())
dimension_H1 = pdsql(q3_h1, locals())
dimension_H3 = pdsql(q3_h3, locals())
dimension_H5 = pdsql(q3_h5, locals())
dimension_H10 = pdsql(q3_h10, locals())
dimension_MRank = pdsql(q3_mr, locals())

q3_cyc_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY dimension"""
q3_mrr_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY dimension"""
q3_h1_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY dimension"""
q3_h3_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY dimension"""
q3_h5_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY dimension"""
q3_h10_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY dimension"""
q3_mr_all = """SELECT dimension, COUNT(dimension) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY dimension"""

dimension_cycle_all = pdsql(q3_cyc_all, locals())
dimension_MRR_all = pdsql(q3_mrr_all, locals())
dimension_H1_all = pdsql(q3_h1_all, locals())
dimension_H3_all = pdsql(q3_h3_all, locals())
dimension_H5_all = pdsql(q3_h5_all, locals())
dimension_H10_all = pdsql(q3_h10_all, locals())
dimension_MRank_all = pdsql(q3_mr_all, locals())

################################################################################
################################## q4: LType ###################################
################################################################################

q4_cyc = """
    SELECT b.LType, COUNT(b.LType) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, learnFac,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.LType
    ;
    """
q4_mrr = """
    SELECT b.LType, COUNT(b.LType) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.LType
    ;
    """
q4_h1 = """
    SELECT b.LType, COUNT(b.LType) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.LType
    ;
    """
q4_h3 = """
    SELECT b.LType, COUNT(b.LType) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.LType
    ;
    """
q4_h5 = """
    SELECT b.LType, COUNT(b.LType) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.LType
    ;
    """
q4_h10 = """
    SELECT b.LType, COUNT(b.LType) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.LType
    ;
    """
q4_mr = """
    SELECT b.LType, COUNT(b.LType) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.LType
    ;
    """

LType_cycle = pdsql(q4_cyc, locals())
LType_MRR = pdsql(q4_mrr, locals())
LType_H1 = pdsql(q4_h1, locals())
LType_H3 = pdsql(q4_h3, locals())
LType_H5 = pdsql(q4_h5, locals())
LType_H10 = pdsql(q4_h10, locals())
LType_MRank = pdsql(q4_mr, locals())

q4_cyc_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY LType"""
q4_mrr_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY LType"""
q4_h1_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY LType"""
q4_h3_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY LType"""
q4_h5_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY LType"""
q4_h10_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY LType"""
q4_mr_all = """SELECT LType, COUNT(LType) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY LType"""

LType_cycle_all = pdsql(q4_cyc_all, locals())
LType_MRR_all = pdsql(q4_mrr_all, locals())
LType_H1_all = pdsql(q4_h1_all, locals())
LType_H3_all = pdsql(q4_h3_all, locals())
LType_H5_all = pdsql(q4_h5_all, locals())
LType_H10_all = pdsql(q4_h10_all, locals())
LType_MRank_all = pdsql(q4_mr_all, locals())

################################################################################
################################# q5: learnFac #################################
################################################################################

q5_cyc = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, margin, constr, LType,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, ABType, dimension, margin, constr, LType) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.margin = b.margin
        AND a.constr = b.constr
        AND a.LType = b.LType
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.learnFac
    ;
    """
q5_mrr = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.learnFac
    ;
    """
q5_h1 = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.learnFac
    ;
    """
q5_h3 = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.learnFac
    ;
    """
q5_h5 = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.learnFac
    ;
    """
q5_h10 = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.learnFac
    ;
    """
q5_mr = """
    SELECT b.learnFac, COUNT(b.learnFac) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.learnFac
    ;
    """

learnFac_cycle = pdsql(q5_cyc, locals())
learnFac_MRR = pdsql(q5_mrr, locals())
learnFac_H1 = pdsql(q5_h1, locals())
learnFac_H3 = pdsql(q5_h3, locals())
learnFac_H5 = pdsql(q5_h5, locals())
learnFac_H10 = pdsql(q5_h10, locals())
learnFac_MRank = pdsql(q5_mr, locals())

q5_cyc_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY learnFac"""
q5_mrr_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY learnFac"""
q5_h1_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY learnFac"""
q5_h3_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY learnFac"""
q5_h5_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY learnFac"""
q5_h10_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY learnFac"""
q5_mr_all = """SELECT learnFac, COUNT(learnFac) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY learnFac"""

learnFac_cycle_all = pdsql(q5_cyc_all, locals())
learnFac_MRR_all = pdsql(q5_mrr_all, locals())
learnFac_H1_all = pdsql(q5_h1_all, locals())
learnFac_H3_all = pdsql(q5_h3_all, locals())
learnFac_H5_all = pdsql(q5_h5_all, locals())
learnFac_H10_all = pdsql(q5_h10_all, locals())
learnFac_MRank_all = pdsql(q5_mr_all, locals())

################################################################################
################################## q6: margin ##################################
################################################################################
q6_cyc = """
    SELECT b.margin, COUNT(b.margin) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, constr, learnFac,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, ABType, dimension, LType, constr, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.constr = b.constr
        AND a.learnFac = b.learnFac
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.margin
    ;
    """
q6_mrr = """
    SELECT b.margin, COUNT(b.margin) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.margin
    ;
    """
q6_h1 = """
    SELECT b.margin, COUNT(b.margin) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.margin
    ;
    """
q6_h3 = """
    SELECT b.margin, COUNT(b.margin) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.margin
    ;
    """
q6_h5 = """
    SELECT b.margin, COUNT(b.margin) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.margin
    ;
    """
q6_h10 = """
    SELECT b.margin, COUNT(b.margin) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.margin
    ;
    """
q6_mr = """
    SELECT b.margin, COUNT(b.margin) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.margin
    ;
    """

margin_cycle = pdsql(q6_cyc, locals())
margin_MRR = pdsql(q6_mrr, locals())
margin_H1 = pdsql(q6_h1, locals())
margin_H3 = pdsql(q6_h3, locals())
margin_H5 = pdsql(q6_h5, locals())
margin_H10 = pdsql(q6_h10, locals())
margin_MRank = pdsql(q6_mr, locals())

q6_cyc_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY margin"""
q6_mrr_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY margin"""
q6_h1_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY margin"""
q6_h3_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY margin"""
q6_h5_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY margin"""
q6_h10_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY margin"""
q6_mr_all = """SELECT margin, COUNT(margin) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY margin"""

margin_cycle_all = pdsql(q6_cyc_all, locals())
margin_MRR_all = pdsql(q6_mrr_all, locals())
margin_H1_all = pdsql(q6_h1_all, locals())
margin_H3_all = pdsql(q6_h3_all, locals())
margin_H5_all = pdsql(q6_h5_all, locals())
margin_H10_all = pdsql(q6_h10_all, locals())
margin_MRank_all = pdsql(q6_mr_all, locals())

################################################################################
################################## q7: constr ##################################
################################################################################

q7_cyc = """
    SELECT b.constr, COUNT(b.constr) AS BestCycCount, AVG(MinCycle) AS AvgCycOfBest
    FROM df b
    INNER JOIN (SELECT
                    HType, ABType, dimension, LType, margin, learnFac,
                    MIN(convergeCycle) AS MinCycle
                FROM df
                GROUP BY HType, ABType, dimension, LType, margin, learnFac) a ON
        a.HType = b.HType
        AND a.ABType = b.ABType
        AND a.dimension = b.dimension
        AND a.LType = b.LType
        AND a.margin = b.margin
        AND a.learnFac = b.learnFac
        AND a.MinCycle = b.convergeCycle
    GROUP BY b.constr
    ;
    """
q7_mrr = """
    SELECT b.constr, COUNT(b.constr) AS BestMRRCount, AVG(MaxMRR) AS AvgMRROfBest, MAX(MaxMRR) AS BestMRROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMRR
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
    GROUP BY b.constr
    ;
    """
q7_h1 = """
    SELECT b.constr, COUNT(b.constr) AS BestH1Count, AVG(MaxHITS1) AS AvgH1OfBest, MAX(MaxHITS1) AS BestH1OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH1
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
    GROUP BY b.constr
    ;
    """
q7_h3 = """
    SELECT b.constr, COUNT(b.constr) AS BestH3Count, AVG(MaxHITS3) AS AvgH3OfBest, MAX(MaxHITS3) AS BestH3OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH3
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
    GROUP BY b.constr
    ;
    """
q7_h5 = """
    SELECT b.constr, COUNT(b.constr) AS BestH5Count, AVG(MaxHITS5) AS AvgH5OfBest, MAX(MaxHITS5) AS BestH5OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH5
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
    GROUP BY b.constr
    ;
    """
q7_h10 = """
    SELECT b.constr, COUNT(b.constr) AS BestH10Count, AVG(MaxHITSX) AS AvgH10OfBest, MAX(MaxHITSX) AS BestH10OfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestH10
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
    GROUP BY b.constr
    ;
    """
q7_mr = """
    SELECT b.constr, COUNT(b.constr) AS BestMRCount, AVG(MinMRank) AS AvgMROfBest, MIN(MinMRank) AS BestMROfBest, ROUND(AVG(b.convergeCycle)) AS AvgCycBestMR
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
    GROUP BY b.constr
    ;
    """

constr_cycle = pdsql(q7_cyc, locals())
constr_MRR = pdsql(q7_mrr, locals())
constr_H1 = pdsql(q7_h1, locals())
constr_H3 = pdsql(q7_h3, locals())
constr_H5 = pdsql(q7_h5, locals())
constr_H10 = pdsql(q7_h10, locals())
constr_MRank = pdsql(q7_mr, locals())

q7_cyc_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(convergeCycle) AS AvgCycOfTotal FROM df GROUP BY constr"""
q7_mrr_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(mrr) AS AvgMRROfTotal, MAX(mrr) AS BestMRROfTotal FROM df GROUP BY constr"""
q7_h1_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(hitsAt1) AS AvgH1OfTotal, MAX(hitsAt1) AS BestH1OfTotal FROM df GROUP BY constr"""
q7_h3_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(hitsAt3) AS AvgH3OfTotal, MAX(hitsAt3) AS BestH3OfTotal FROM df GROUP BY constr"""
q7_h5_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(hitsAt5) AS AvgH5OfTotal, MAX(hitsAt5) AS BestH5OfTotal FROM df GROUP BY constr"""
q7_h10_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(hitsAtX) AS AvgH10OfTotal, MAX(hitsAtX) AS BestH10OfTotal FROM df GROUP BY constr"""
q7_mr_all = """SELECT constr, COUNT(constr) AS TotalCount, AVG(mrank) AS AvgMROfTotal, MIN(mrank) AS BestMROfTotal FROM df GROUP BY constr"""

constr_cycle_all = pdsql(q7_cyc_all, locals())
constr_MRR_all = pdsql(q7_mrr_all, locals())
constr_H1_all = pdsql(q7_h1_all, locals())
constr_H3_all = pdsql(q7_h3_all, locals())
constr_H5_all = pdsql(q7_h5_all, locals())
constr_H10_all = pdsql(q7_h10_all, locals())
constr_MRank_all = pdsql(q7_mr_all, locals())

################################################################################
################################ DATA ASSEMBLY #################################
################################################################################

q1_table = """
    SELECT d.HType, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM HType_cycle_all d
    LEFT JOIN HType_cycle c         ON c.HType = d.HType
    LEFT JOIN HType_MRR e           ON d.HType = e.HType
    LEFT JOIN HType_MRR_all f       ON d.HType = f.HType
    LEFT JOIN HType_H1 g            ON d.HType = g.HType
    LEFT JOIN HType_H1_all h        ON d.HType = h.HType
    LEFT JOIN HType_H3 i            ON d.HType = i.HType
    LEFT JOIN HType_H3_all j        ON d.HType = j.HType
    LEFT JOIN HType_H5 k            ON d.HType = k.HType
    LEFT JOIN HType_H5_all l        ON d.HType = l.HType
    LEFT JOIN HType_H10 m           ON d.HType = m.HType
    LEFT JOIN HType_H10_all n       ON d.HType = n.HType
    LEFT JOIN HType_MRank o         ON d.HType = o.HType
    LEFT JOIN HType_MRank_all p     ON d.HType = p.HType
    """
q2_table = """
    SELECT d.ABType, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM ABType_cycle_all d
    LEFT JOIN ABType_cycle c         ON c.ABType = d.ABType
    LEFT JOIN ABType_MRR e           ON d.ABType = e.ABType
    LEFT JOIN ABType_MRR_all f       ON d.ABType = f.ABType
    LEFT JOIN ABType_H1 g            ON d.ABType = g.ABType
    LEFT JOIN ABType_H1_all h        ON d.ABType = h.ABType
    LEFT JOIN ABType_H3 i            ON d.ABType = i.ABType
    LEFT JOIN ABType_H3_all j        ON d.ABType = j.ABType
    LEFT JOIN ABType_H5 k            ON d.ABType = k.ABType
    LEFT JOIN ABType_H5_all l        ON d.ABType = l.ABType
    LEFT JOIN ABType_H10 m           ON d.ABType = m.ABType
    LEFT JOIN ABType_H10_all n       ON d.ABType = n.ABType
    LEFT JOIN ABType_MRank o         ON d.ABType = o.ABType
    LEFT JOIN ABType_MRank_all p     ON d.ABType = p.ABType
    """
q3_table = """
    SELECT d.dimension, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM dimension_cycle_all d
    LEFT JOIN dimension_cycle c         ON c.dimension = d.dimension
    LEFT JOIN dimension_MRR e           ON d.dimension = e.dimension
    LEFT JOIN dimension_MRR_all f       ON d.dimension = f.dimension
    LEFT JOIN dimension_H1 g            ON d.dimension = g.dimension
    LEFT JOIN dimension_H1_all h        ON d.dimension = h.dimension
    LEFT JOIN dimension_H3 i            ON d.dimension = i.dimension
    LEFT JOIN dimension_H3_all j        ON d.dimension = j.dimension
    LEFT JOIN dimension_H5 k            ON d.dimension = k.dimension
    LEFT JOIN dimension_H5_all l        ON d.dimension = l.dimension
    LEFT JOIN dimension_H10 m           ON d.dimension = m.dimension
    LEFT JOIN dimension_H10_all n       ON d.dimension = n.dimension
    LEFT JOIN dimension_MRank o         ON d.dimension = o.dimension
    LEFT JOIN dimension_MRank_all p     ON d.dimension = p.dimension
    """
q4_table = """
    SELECT d.LType, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM LType_cycle_all d
    LEFT JOIN LType_cycle c         ON c.LType = d.LType
    LEFT JOIN LType_MRR e           ON d.LType = e.LType
    LEFT JOIN LType_MRR_all f       ON d.LType = f.LType
    LEFT JOIN LType_H1 g            ON d.LType = g.LType
    LEFT JOIN LType_H1_all h        ON d.LType = h.LType
    LEFT JOIN LType_H3 i            ON d.LType = i.LType
    LEFT JOIN LType_H3_all j        ON d.LType = j.LType
    LEFT JOIN LType_H5 k            ON d.LType = k.LType
    LEFT JOIN LType_H5_all l        ON d.LType = l.LType
    LEFT JOIN LType_H10 m           ON d.LType = m.LType
    LEFT JOIN LType_H10_all n       ON d.LType = n.LType
    LEFT JOIN LType_MRank o         ON d.LType = o.LType
    LEFT JOIN LType_MRank_all p     ON d.LType = p.LType
    """
q5_table = """
    SELECT d.learnFac, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM learnFac_cycle_all d
    LEFT JOIN learnFac_cycle c         ON c.learnFac = d.learnFac
    LEFT JOIN learnFac_MRR e           ON d.learnFac = e.learnFac
    LEFT JOIN learnFac_MRR_all f       ON d.learnFac = f.learnFac
    LEFT JOIN learnFac_H1 g            ON d.learnFac = g.learnFac
    LEFT JOIN learnFac_H1_all h        ON d.learnFac = h.learnFac
    LEFT JOIN learnFac_H3 i            ON d.learnFac = i.learnFac
    LEFT JOIN learnFac_H3_all j        ON d.learnFac = j.learnFac
    LEFT JOIN learnFac_H5 k            ON d.learnFac = k.learnFac
    LEFT JOIN learnFac_H5_all l        ON d.learnFac = l.learnFac
    LEFT JOIN learnFac_H10 m           ON d.learnFac = m.learnFac
    LEFT JOIN learnFac_H10_all n       ON d.learnFac = n.learnFac
    LEFT JOIN learnFac_MRank o         ON d.learnFac = o.learnFac
    LEFT JOIN learnFac_MRank_all p     ON d.learnFac = p.learnFac
    """
q6_table = """
    SELECT d.margin, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM margin_cycle_all d
    LEFT JOIN margin_cycle c         ON c.margin = d.margin
    LEFT JOIN margin_MRR e           ON d.margin = e.margin
    LEFT JOIN margin_MRR_all f       ON d.margin = f.margin
    LEFT JOIN margin_H1 g            ON d.margin = g.margin
    LEFT JOIN margin_H1_all h        ON d.margin = h.margin
    LEFT JOIN margin_H3 i            ON d.margin = i.margin
    LEFT JOIN margin_H3_all j        ON d.margin = j.margin
    LEFT JOIN margin_H5 k            ON d.margin = k.margin
    LEFT JOIN margin_H5_all l        ON d.margin = l.margin
    LEFT JOIN margin_H10 m           ON d.margin = m.margin
    LEFT JOIN margin_H10_all n       ON d.margin = n.margin
    LEFT JOIN margin_MRank o         ON d.margin = o.margin
    LEFT JOIN margin_MRank_all p     ON d.margin = p.margin
    """
q7_table = """
    SELECT d.constr, d.AvgCycOfTotal,
                    e.BestMRRCount, e.AvgMRROfBest, f.AvgMRROfTotal, e.BestMRROfBest, f.BestMRROfTotal, e.AvgCycBestMRR,
                    g.BestH1Count,  g.AvgH1OfBest,  h.AvgH1OfTotal, g.BestH1OfBest, h.BestH1OfTotal, g.AvgCycBestH1,
                    i.BestH3Count,  i.AvgH3OfBest,  j.AvgH3OfTotal, i.BestH3OfBest, j.BestH3OfTotal, i.AvgCycBestH3,
                    k.BestH5Count,  k.AvgH5OfBest,  l.AvgH5OfTotal, k.BestH5OfBest, l.BestH5OfTotal, k.AvgCycBestH5,
                    m.BestH10Count, m.AvgH10OfBest, n.AvgH10OfTotal, m.BestH10OfBest, n.BestH10OfTotal, m.AvgCycBestH10,
                    o.BestMRCount,  o.AvgMROfBest,  p.AvgMROfTotal, o.BestMROfBest, p.BestMROfTotal, o.AvgCycBestMR
    FROM constr_cycle_all d
    LEFT JOIN constr_cycle c         ON c.constr = d.constr
    LEFT JOIN constr_MRR e           ON d.constr = e.constr
    LEFT JOIN constr_MRR_all f       ON d.constr = f.constr
    LEFT JOIN constr_H1 g            ON d.constr = g.constr
    LEFT JOIN constr_H1_all h        ON d.constr = h.constr
    LEFT JOIN constr_H3 i            ON d.constr = i.constr
    LEFT JOIN constr_H3_all j        ON d.constr = j.constr
    LEFT JOIN constr_H5 k            ON d.constr = k.constr
    LEFT JOIN constr_H5_all l        ON d.constr = l.constr
    LEFT JOIN constr_H10 m           ON d.constr = m.constr
    LEFT JOIN constr_H10_all n       ON d.constr = n.constr
    LEFT JOIN constr_MRank o         ON d.constr = o.constr
    LEFT JOIN constr_MRank_all p     ON d.constr = p.constr
    """

HType_table = pdsql(q1_table, locals())
ABType_table = pdsql(q2_table, locals())
dimension_table = pdsql(q3_table, locals())
LType_table = pdsql(q4_table, locals())
learnFac_table = pdsql(q5_table, locals())
margin_table = pdsql(q6_table, locals())
constr_table = pdsql(q7_table, locals())


tables = [HType_table, ABType_table, dimension_table, LType_table, learnFac_table, margin_table, constr_table]
with open(output, "w") as csv_stream:
    for table in tables:
        table.to_csv(csv_stream, index=False)
        csv_stream.write("\n")

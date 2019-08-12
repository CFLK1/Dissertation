################################################################################
# a Random Forest regressor
# analysis by feature importances and partial dependence plot
# note that codes from pdpbox 0.2.0 have been modified
# in pcp_plot_utils.py line 252, contour_label_fontsize is changed to fontsize,
# but it can no longer draw individual feature plots
# For the correct use of bars for categorical features,
# line 197, 198 are commented out, replaced by:
# ax.bar(x, y, color='#777777', yerr=std, align='center')
# line 200, 201 on std_fill are commented out
# line 203 remove *2 for both min and max
# in pcp.py, to generate a customised plot the following lanes are commented out
# 379, 382-386
# 418 has "outer_grid[1]" argument removed
# 450 has "title_ax" key-value pair removed
################################################################################

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from pdpbox import pdp



def HP_regress(data, target, outdir, dataset, ABType=True):

################################################################################
# preprocessing
################################################################################

    df = pd.read_csv(data, index_col=0)

    X = df.iloc[:,:7]
    y = df[target]

    if ABType == True:
        #ohe = OneHotEncoder()
        #X = ohe.fit_transform(X)
        ore = OrdinalEncoder(categories=[["H1","H2","H3","H4"],["A","B"],\
        [8,16,32,64],[10,100,1000],[1,2,4],["fixed","max"],["L0","L2","L1"]])
        X = ore.fit_transform(X)

    else:
        X = X.drop(columns=["ABType"])
        #ohe = OneHotEncoder()
        #X = ohe.fit_transform(X)
        ore = OrdinalEncoder(categories=[["H1","H2","H3","H4"],\
        [8,16,32,64],[10,100,1000],[1,2,4],["fixed","max"],["L0","L2","L1"]])
        X = ore.fit_transform(X)

################################################################################
# RF
################################################################################

    gsc = GridSearchCV(estimator=RandomForestRegressor(),\
    param_grid={"max_depth": range(5,11), \
    "n_estimators": (500, 1000)},\
    cv=5, scoring="neg_mean_absolute_error", verbose=0, n_jobs=-1)

    gs_result = gsc.fit(X, y)
    best_params = gs_result.best_params_
    print(best_params)

    rfr = RandomForestRegressor(max_depth=best_params["max_depth"],\
    n_estimators=best_params["n_estimators"], random_state=42)

    cv_scores = cross_val_score(rfr, X, y, cv=10, \
    scoring="neg_mean_absolute_error")

    rfr.fit(X, y)

################################################################################
# feature importances
################################################################################

    importances = rfr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfr.estimators_],\
    axis=0)
    imp_data = [tree.feature_importances_ for tree in rfr.estimators_]
    #ci = stats.sem(imp_data) * stats.t.ppf(1.95/2., len(imp_data)-1)

    indices = np.argsort(importances)[::-1]

    if ABType == True:
        xlab = np.where(indices==0, "ont. constr.", indices)
        xlab = np.where(xlab=="1", "rel. norm.", xlab)
        xlab = np.where(xlab=="2", "dimension", xlab)
        xlab = np.where(xlab=="3", "learn. rate", xlab)
        xlab = np.where(xlab=="4", "margin", xlab)
        xlab = np.where(xlab=="5", "magnitude", xlab)
        xlab = np.where(xlab=="6", "method", xlab)
    else:
        xlab = np.where(indices==0, "ont. constr.", indices)
        xlab = np.where(xlab=="1", "dimension", xlab)
        xlab = np.where(xlab=="2", "learn. rate", xlab)
        xlab = np.where(xlab=="3", "margin", xlab)
        xlab = np.where(xlab=="4", "magnitude", xlab)
        xlab = np.where(xlab=="5", "method", xlab)

    fig1, ax1 = plt.subplots(figsize=(3.6,3.6))
    ax1.set_title("10-fold CV NMAE = %.2f, std = %.2f" % (np.mean(cv_scores), np.std(cv_scores)), size=10)
    ax1.bar(range(X.shape[1]), importances[indices], color="#777777",\
    yerr=std[indices], align="center")
    ax1.set_xticks(list(range(X.shape[1])))
    ax1.set_xticklabels(xlab, size=9)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_xlim([-1, X.shape[1]])
    ax1.set_ylabel("relative feature importance")
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=":", color="#777777")
    fig1.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

################################################################################
# partial dependence plot
################################################################################

    X_df = pd.DataFrame(X)
    df2 = pd.concat([X_df, y], axis=1, sort=False)
    if ABType == True:
        df2.columns = ["HType", "ABType", "dimension", "learnFac", "margin", "constr", "LType", target]
        mod_feats = ["HType", "ABType", "dimension", "learnFac", "margin", "constr", "LType"]
    else:
        df2.columns = ["HType", "dimension", "learnFac", "margin", "constr", "LType", target]
        mod_feats = ["HType", "dimension", "learnFac", "margin", "constr", "LType"]

    # HType

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="HType")

    fig2, ax2 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="ontology constraints", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax2["pdp_ax"].set_xticklabels(["H+T", "H+TI", "H+TID", "H+TIDF"])
    ax2["pdp_ax"].set_xticks([0, 1, 2, 3])
    if target is not "mrank":
        ax2["pdp_ax"].set_ylim([0, 1])
    ax2["pdp_ax"].set_ylabel("predicted metric score")
    ax2["pdp_ax"].grid(color="#777777")
    fig2.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # dimension

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="dimension")

    fig3, ax3 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="dimension $k$", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax3["pdp_ax"].set_xticklabels(["8", "16", "32", "64"])
    ax3["pdp_ax"].set_xticks([0, 1, 2, 3])
    if target is not "mrank":
        ax3["pdp_ax"].set_ylim([0, 1])
    ax3["pdp_ax"].set_ylabel("predicted metric score")
    ax3["pdp_ax"].grid(color="#777777")
    fig3.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # learnFac

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="learnFac", \
    num_grid_points=3)

    fig4, ax4 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="learning rate \u03BB", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax4["pdp_ax"].set_xticklabels(["0.1", "0.01", "0.001"])
    ax4["pdp_ax"].set_xticks([0, 1, 2])
    if target is not "mrank":
        ax4["pdp_ax"].set_ylim([0, 1])
    ax4["pdp_ax"].set_ylabel("predicted metric score")
    ax4["pdp_ax"].grid(color="#777777")
    fig4.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # margin

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="margin", \
    num_grid_points=3)

    fig5, ax5 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="margin $\gamma$", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax5["pdp_ax"].set_xticklabels(["1", "2", "4"])
    ax5["pdp_ax"].set_xticks([0, 1, 2])
    if target is not "mrank":
        ax5["pdp_ax"].set_ylim([0, 1])
    ax5["pdp_ax"].set_ylabel("predicted metric score")
    ax5["pdp_ax"].grid(color="#777777")
    fig5.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # constr

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="constr")

    fig6, ax6 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="regularisation magnitude", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax6["pdp_ax"].set_xticklabels(["surface", "space"])
    ax6["pdp_ax"].set_xticks([0, 1])
    if target is not "mrank":
        ax6["pdp_ax"].set_ylim([0, 1])
    ax6["pdp_ax"].set_ylabel("predicted metric score")
    ax6["pdp_ax"].grid(color="#777777")
    fig6.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # LType

    pdp_rfr = pdp.pdp_isolate(model=rfr, \
    dataset=df2, \
    model_features=mod_feats, \
    feature="LType", \
    num_grid_points=3)

    fig7, ax7 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
    feature_name="training method", \
    center=False, plot_lines=False, \
    plot_pts_dist=False, figsize = (3.6,3.6))
    ax7["pdp_ax"].set_xticklabels(["linear", "projection", "hybrid"])
    ax7["pdp_ax"].set_xticks([0, 1, 2])
    if target is not "mrank":
        ax7["pdp_ax"].set_ylim([0, 1])
    ax7["pdp_ax"].set_ylabel("predicted metric score")
    ax7["pdp_ax"].grid(color="#777777")
    fig7.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
    hspace=0.2, wspace=0.2)

    # ABType (if true)

    if ABType == True:
        pdp_rfr = pdp.pdp_isolate(model=rfr, \
        dataset=df2, \
        model_features=mod_feats, \
        feature="ABType")

        fig8, ax8 = pdp.pdp_plot(pdp_isolate_out=pdp_rfr,
        feature_name="relation normalisation", \
        center=False, plot_lines=False, \
        plot_pts_dist=False, figsize = (3.6,3.6))
        ax8["pdp_ax"].set_xticklabels(["False", "True"])
        ax8["pdp_ax"].set_xticks([0, 1])
        if target is not "mrank":
            ax8["pdp_ax"].set_ylim([0, 1])
        ax8["pdp_ax"].set_ylabel("predicted metric score")
        ax8["pdp_ax"].grid(color="#777777")
        fig8.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, \
        hspace=0.2, wspace=0.2)

    #plt.show()

################################################################################
# save figures
################################################################################

    subdir = "/".join([outdir, dataset, target])
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    fig1_path = "/".join([subdir, "_".join([dataset, target, "RF_importance.png"])])
    fig1.savefig(fig1_path)

    fig2_path = "/".join([subdir, "_".join([dataset, target, "PDP_HType.png"])])
    fig2.savefig(fig2_path)

    fig3_path = "/".join([subdir, "_".join([dataset, target, "PDP_dimension.png"])])
    fig3.savefig(fig3_path)

    fig4_path = "/".join([subdir, "_".join([dataset, target, "PDP_learnFac.png"])])
    fig4.savefig(fig4_path)

    fig5_path = "/".join([subdir, "_".join([dataset, target, "PDP_margin.png"])])
    fig5.savefig(fig5_path)

    fig6_path = "/".join([subdir, "_".join([dataset, target, "PDP_constr.png"])])
    fig6.savefig(fig6_path)

    fig7_path = "/".join([subdir, "_".join([dataset, target, "PDP_LType.png"])])
    fig7.savefig(fig7_path)

    if ABType == True:
        fig8_path = "/".join([subdir, "_".join([dataset, target, "PDP_ABType.png"])])
        fig8.savefig(fig8_path)

################################################################################

# data:         path of data (jointable)
# target:       string of target column name (metric) (mrr, mrank, MLP_acc, etc)
# outdir:       path for outputting folders and files
# dataset:      dataset name (only determines name of created folder and files)
# ABType:       whether "B" is an option

datasets = ["FULL", "N2N", "PROC", "PTREEv2", "IHDv2", "IHPv2", "MUSHv2"]
targets = ["mrr", "mrank", "MLP_acc", "MLP_wF1"]

for i, dataset in enumerate(datasets):

    data = "./norange/new_remain/" + dataset + "_jointable_norange.csv"

    for target in targets:
        HP_regress(data=data, \
        target=target, \
        outdir="./norange/new_remain", \
        dataset=dataset, \
        ABType=True if i < 4 else False)

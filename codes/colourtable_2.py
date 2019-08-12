import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def coltab2(ds, metric, ABType=True):

    dsnames = {"FULL":"BPA-Full", "N2N":"BPA-Large", "PROC":"BPA-Medium", \
    "PTREEv2":"BPA-Small", "IHDv2":"Demographics", "IHPv2":"Pregnancy", \
    "MUSHv2":"Mushroom"}
    title = dsnames[ds]

    input = "./norange/new_remain/" + ds + "_jointable_norange.csv"
    data = pd.read_csv(input, index_col=0)

    metricnames = {"mrr":"MRR", "MLP_acc":"MLP accuracy", \
    "MLP_wF1":"MLP F1 score"}
    metriclab = metricnames[metric]

    output = "./norange/heatmaps/" + ds + "_" + metric + "_colourtab2.png"

################################################################################
################################## for all hyp ###################################
################################################################################

    data = data.rename(index=str, columns={"HType":"ont. constr.", \
    "ABType":"rel. norm.", "dimension":"k", "learnFac":"\u03BB", \
    "margin":"\u03B3", "constr":"magnitude", "LType":"method"})

    if ABType:
        hyperparameters = ["ont. constr.", "rel. norm.", "k", "\u03BB", "\u03B3", \
        "magnitude", "method"]
    else:
        hyperparameters = ["ont. constr.", "k", "\u03BB", "\u03B3", \
        "magnitude", "method"]

    cmap = sns.cubehelix_palette(50, gamma=2.5, hue=0, rot=0, light=0.98, dark=0, as_cmap=True)

    if ABType:
        f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7, \
        gridspec_kw={"width_ratios":[4,2,4,3,3,2,3]}, figsize=(6, 3))
        axcb = f.add_axes([.89, .3, .02, .4])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        ax1.get_shared_y_axes().join(ax2, ax3, ax4, ax5, ax6, ax7)
    else:
        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, \
        gridspec_kw={"width_ratios":[4,4,3,3,2,3]}, figsize=(6, 3))
        axcb = f.add_axes([.89, .3, .02, .4])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        ax1.get_shared_y_axes().join(ax2, ax3, ax4, ax5, ax6)

    for i, hyperparameter in enumerate(hyperparameters):

        hp = data.filter(items=[hyperparameter, metric])

        cut = pd.IntervalIndex.from_breaks(np.arange(0,105,5)/100)
        labels = np.arange(0,100,5)/100
        hp[metriclab] = pd.cut(hp[metric], bins=cut)

        df = hp.drop(metric, axis=1).groupby([metriclab, hyperparameter]).size().unstack().fillna(0)
        df = df.reindex(cut, fill_value=0).iloc[::-1]
        for col in df:
            df[col] = df[col].values / np.sum(df[col].values)
            # update value (col) names
        if hyperparameter == "ont. constr.":
            df.columns = ["H+T", "H+TI", "H+TID", "H+TIDF"]
        if hyperparameter == "rel. norm.":
            df.columns = ["norm.-F", "norm.-T"]
        if hyperparameter == "k":
            df.columns = ["$k$-8", "$k$-16", "$k$-32", "$k$-64"]
        if hyperparameter == "\u03BB":
            df.columns = ["\u03BB-0.1", "\u03BB-0.01", "\u03BB-0.001"]
        if hyperparameter == "\u03B3":
            df.columns = ["\u03B3-1", "\u03B3-2", "\u03B3-4"]
        if hyperparameter == "magnitude":
            df.columns = ["surface", "space"]
        if hyperparameter == "method":
            df.columns = ["linear", "hybrid", "proj."]

        if i == len(hyperparameters)-1:
            g = sns.heatmap(df, vmin=0, vmax=1, \
            cmap=cmap, ax=axes[i], cbar_ax=axcb, \
            cbar_kws={"label":"proportion of models in interval"}) #last graph
            tlx = g.get_xticklabels()
            g.set_xticklabels(tlx, rotation=90, size=9)
            tly = g.get_yticklabels()
            g.set_yticklabels(tly, size=9)
            # color bar
            cbtly = g.figure.axes[-1].get_yticklabels()
            g.figure.axes[-1].set_yticklabels(cbtly, size=9)
        else:
            g = sns.heatmap(df, vmin=0, vmax=1, \
            cmap=cmap, cbar=False, ax=axes[i])
            tlx = g.get_xticklabels()
            g.set_xticklabels(tlx, rotation=90, size=9)
            tly = g.get_yticklabels()
            g.set_yticklabels(tly, size=9)

        g.set_xlabel("")
        if i:
            g.set_ylabel("")
            g.set_yticks([])
        else:
            ylab = metriclab + " intervals"
            g.set_ylabel(ylab)

    #plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, .9, .95])
    plt.subplots_adjust(wspace=0.1)

    plt.savefig(output)

################################################################################

# ds:       dataset [FULL, N2N, PROC, PTREEv2, IHDv2, IHPv2, MUSHv2]
# metric:   metric name in input csv, supports [mrr, MLP_acc, MLP_wF1]
# ABType:   Does ds have relation normalisation option? (default True)

dss = ["FULL", "N2N", "PROC", "PTREEv2", "IHDv2", "IHPv2", "MUSHv2"]
metrics = ["mrr", "MLP_acc", "MLP_wF1"]

for i, ds in enumerate(dss):
    for metric in metrics:
        coltab2(ds=ds, \
        metric=metric, \
        ABType=True if i < 4 else False)

"""
Functions for constructing data visualizations
"""
import pandas as pd
import numpy as np
import itertools 
import time
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import stats


def myplots(x, name):
    '''
    Helper function for plotting data visualizations.
    A figure is generated that consists of a sequence plot,
    a histogram, a boxplot, and a probability plot
    PARAMS:
        x: a single feature column
        name: feature name
    RETURNS:
        fig: the handle to the figure
    '''
    xplt = deepcopy(x)
    #nsamples = xplt.shape[0]
    
    # Compute descriptive statistics
    med = np.nanmedian(xplt)
    mu = np.nanmean(xplt)
    sig = np.nanstd(xplt)
    mn = np.nanmin(xplt)
    mx = np.nanmax(xplt)

    # Plots
    fig = plt.figure(figsize=(14,3))
    
    # Sequence plot
    plt.subplot(1,4,1)
    plt.plot(xplt)
    plt.hlines(mu, 0, xplt.shape[0], colors='r', linestyles='dashed', label='mean')
    #plt.xlabel("Sample Index")
    plt.ylabel(name)

    # Histogram
    plt.subplot(1,4,2)
    (n, b, p) = plt.hist(xplt)
    plt.plot((mu, mu), (0, np.max(n)), 'r--')
    plt.xlabel(name)
    
    # Boxplot
    plt.subplot(1,4,3)
    plt.boxplot(xplt, labels=[name])
    plt.ylabel(name)
    plt.tight_layout()
    
    # Probability Plot
    subplt = plt.subplot(1,4,4)
    (osm, osr), (slope, intercept, r) = stats.probplot(xplt, dist='norm', plot=subplt)
    plt.tight_layout()
    plt.show()
    
    #print("myplots", name)
    return fig
    
def featureplots(X, feature_names):
    '''
    Call myplots for all the features
    PARAMS:
        X: full data set
        feature_names: list of the feature names for the columns
    '''
    for f, feature_name in enumerate(feature_names):
        print("FEATURE:", feature_name)
        x = X[:, f]
        myplots(x, feature_name)

def scatter_corrplots(X, feature_names, corrfmt="%.3f", FIGW=15):
    '''
    Construct scatter matrix of feature correlations
    PARAMS:
        X: full data set
        feature_names: list of the feature names for the columns
        corrfmt: string format for the correlation when printed on the plots
    '''
    ncorrs = len(feature_names)
    corrs = np.corrcoef(X.T)
    thresh = .6

    fig, axs = plt.subplots(nrows=ncorrs, ncols=ncorrs, figsize=(FIGW, FIGW))
    for f1, f1_name in enumerate(feature_names):
        for f2, f2_name in enumerate(feature_names):
            # Correlation colorimage
            if f1 < f2:
                cr = corrs[f1, f2]
                im = axs[f1, f2].imshow(np.array([[cr, cr], [cr, cr]]), 
                                        cmap='RdBu', vmin=-1, vmax=1)
                text = axs[f1, f2].text(.5, .5, corrfmt % cr, ha="center", va="center", 
                                        color="w" if abs(cr) > thresh else "k",
                                        fontdict=dict(fontsize=8))
            if f1 == 0 and f2 == (ncorrs - 1):
                cbar = axs[f1, f2].figure.colorbar(im, ax=axs[f1, f2])
                cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

            # Feature histogram
            if f1 == f2:
                axs[f1, f2].hist(X[:, f1], color='green')
            # Feature scatter plot
            if f1 > f2:
                axs[f1, f2].scatter(X[:, f2], X[:, f1], s=1, alpha=.7)

            if ncorrs > 10 or f1 < (ncorrs - 1): axs[f1, f2].set_xticks([])
            if ncorrs > 10 or f2 > 0: axs[f1, f2].set_yticks([])

            if f1 == (ncorrs - 1): axs[f1, f2].set_xlabel(f2_name, rotation=90)
            if f2 == 0: axs[f1, f2].set_ylabel(f1_name, rotation=0)

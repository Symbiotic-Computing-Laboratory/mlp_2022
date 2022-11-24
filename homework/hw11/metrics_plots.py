"""
Metrics and plots of metrics
Evaluate the performance of an already trained model on some data

Based on code from Dr. Fagg's 5970 Machine Learning Practices Lectures

author: Monique Shotande <monique.shotande@ou.edu>
version: Fall 2019
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def skillScore(y_true, y_pred, skill='pss'):
    """
    Compute various skill scores
    PARAMS:
        y_true: the true classification label
        y_pred: the classification predicted by the model (must be binary)
        skill: a string used to select a particular skill score to compute
                'pss' | 'hss' | 'bss'
    """
    cmtx = confusion_matrix(y_true, y_pred)
    tn = cmtx[0,0]
    fp = cmtx[0,1]
    fn = cmtx[1,0]
    tp = cmtx[1,1]

    if skill == 'acc': #accuracy
        return float(tp + tn) / (tp + fn + tn + fp)
    if skill == 'pss':
        tpr = float(tp) / (tp + fn)
        fpr = float(fp) / (fp + tn)
        pss = tpr - fpr
        return  [pss, tpr, fpr] 
    if skill == 'hss': #Heidke
        return 2.0 * (tp*tn - fp*fn) / ((tp+fn) * (fn+tn) + (tp+fp) * (fp+tn))
    if skill == 'bss': #Brier Skill Score
        return np.mean((y_true - y_pred) **2)

def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Under constructions
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    PARAMS:
        cm: the confusion matrix
        classes: list of unique class labels
        normalize: boolean flag whether to normalize values
        title: figure title
        cmap: colormap scheme
    """
    # View percentages
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color='w' if cm[i, j] > thresh else 'k')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_mtx', bbox_inches="tight")

# Generate a color map plot for a confusion matrix
def confusion_mtx_colormap(mtx, xnames, ynames, cbarlabel=""):
    ''' 
    Generate a figure that plots a colormap of a matrix
    PARAMS:
        mtx: matrix of values
        xnames: list of x tick names
        ynames: list of the y tick names
        cbarlabel: label for the color bar
    RETURNS:
        fig, ax: the corresponding handles for the figure and axis
    '''
    nxvars = mtx.shape[1]
    nyvars = mtx.shape[0]

    # create the figure and plot the correlation matrix
    fig, ax = plt.subplots()
    im = ax.imshow(mtx, cmap='summer', zorder=1)
    if not cbarlabel == "":
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Specify the row and column ticks and labels for the figure
    ax.set_xticks(range(nxvars))
    ax.set_yticks(range(nyvars))
    ax.set_xticklabels(xnames)
    ax.set_yticklabels(ynames)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Actual Labels")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, 
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    lbl = np.array([['TN', 'FP'], ['FN', 'TP']])
    for i in range(nyvars):
        for j in range(nxvars):
            thresh = mtx.sum() / 2
            text = ax.text(j, i, "%s = %d" % (lbl[i,j], mtx[i, j]),
                           ha='center', va='center', color='k' if mtx[i, j] > thresh else 'w')
            #text.set_path_effects([peffects.withStroke(linewidth=1.5, 
				#foreground='w' if mtx[i, j] > .5 else 'k' )])

    return fig, ax

# Compute the ROC and PR Curves and generate the KS plot
def ks_roc_prc_plot(targets, scores, FIGWIDTH=15, FIGHEIGHT=6, FONTSIZE=14):
    ''' 
    Generate a figure that plots the ROC and PR Curves and the distributions 
    of the TPR and FPR over a set of thresholds. ROC plots the false alarm rate 
    versus the hit rate. The precision-recall curve (PRC) displays recall vs 
    precision
    PARAMS:
        targets: list of true target labels
        scores: list of predicted scores
    RETURNS:
        roc_results: dict of ROC results: {'tpr', 'fpr', 'thresholds', 'AUC'}
        prc_results: dict of PRC results: {'precision', 'recall', 
                                           'thresholds', 'AUC'}
        fig, axs: corresponding handles for the figure and axis
    '''
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(targets, scores)
    auc_roc = auc(fpr, tpr)

    # Compute precision-recall AUC
    precision, recall, thresholds_prc = precision_recall_curve(targets, scores)
    auc_prc = auc(recall, precision)

    roc_results = {'tpr':tpr, 'fpr':fpr, 'thresholds':thresholds, 'auc':auc_roc}
    prc_results = {'precision':precision, 'recall':recall,
                   'thresholds':thresholds_prc, 'auc':auc_prc}
    #thresholds = {'roc_thres':thresholds, 'prc_thres':thresholds_prc}
    #auc_results = {'roc_auc':auc_roc, 'prc_auc':auc_prc}

    # Compute positve fraction
    pos = np.where(targets)[0]
    npos = targets.sum()
    pos_frac = npos / targets.size

    # Generate KS plot
    fig, ax = plt.subplots(1, 3, figsize=(FIGWIDTH,FIGHEIGHT))
    axs = ax.ravel()
    
    ax[0].plot(thresholds, tpr, color='b')
    ax[0].plot(thresholds, fpr, color='r')
    ax[0].plot(thresholds, tpr - fpr, color='g')
    ax[0].invert_xaxis()
    #ax[0].set_aspect('equal', 'box')
    ax[0].set(xlabel='threshold', ylabel='fraction')
    ax[0].legend(['TPR', 'FPR', 'K-S Distance'], fontsize=FONTSIZE)
    
    # Generate ROC Curve plot
    ax[1].plot(fpr, tpr, color='b')
    ax[1].plot([0,1], [0,1], 'r--')
    ax[1].set(xlabel='FPR', ylabel='TPR')
    ax[1].set_aspect('equal', 'box')
    auc_text = ax[1].text(.05, .95, "AUC = %.4f" % auc_roc, 
                          color="k", fontsize=FONTSIZE)
    #print("ROC AUC:", auc_roc)
    
    # Generate precision-recall Curve plot
    ax[2].plot(recall, precision, color='b')
    ax[2].plot([0, 0, 1], [1, pos_frac, pos_frac], 'r--')
    ax[2].set(xlabel='Recall', ylabel='Precision')
    ax[2].set_aspect('equal', 'box')
    auc_prc_text = plt.text(.2, .95, "PR AUC = %.4f" % auc_prc, 
                            color="k", fontsize=FONTSIZE)
    pos_frac_text = plt.text(.2, .85, "%.2f %% pos" % (pos_frac * 100), 
                             color="k", fontsize=FONTSIZE)
    #print("PRC AUC:", auc_prc)

    return roc_results, prc_results, fig, axs

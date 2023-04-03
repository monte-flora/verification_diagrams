# Plotting dashboard. 

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ._diagrams import VerificationDiagram
from ._curve_utils import sklearn_curve_bootstrap, compute_multiple_curves

def plot_verification(y_true, y_pred, names, style='classification', groups=None, 
                      n_boot=10, figsize=(8,8), table_bbox=None, plot_kwargs={}):
    """Plot Classification- or Regression-based verification."""
    verify = VerificationDiagram()
    fig, axes = plt.subplots(dpi=300, figsize=figsize, ncols=2, nrows=2)
    
    if style == 'classification':
        metrics = ['reliability', 'roc', 'performance']
    elif style == 'regression': 
        metrics = ['taylor']
    
    for ax, metric in zip(axes.flat, metrics):
        xp, yp, pred, scores = compute_multiple_curves(y_true, y_pred, names, groups=groups, 
                         metric=metric, n_boot=n_boot, scorers=None, random_seed=42)
    
        verify.plot(diagram=metric, x=xp, y=yp, ax=ax, scores=scores, 
                    pred=y_pred, 
                    table_bbox=table_bbox, 
                    plot_kwargs=plot_kwargs)
    
    axes.flat[-1].remove()

    return fig, axes
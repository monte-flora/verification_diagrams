# Plotting dashboard. 

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_verification(estimators, X, y, 
                      baseline_estimators=None, 
                      X_baseline=None, 
                      n_boot=10, style='classification'):
    
    """Plot Classification- or Regression-based verification."""
    verify = VerificationDiagram()
    fig, axes = plt.subplots(dpi=300, figsize=(8,8), ncols=2, nrows=2)
    
    if style == 'classification':
        metrics = ['reliability', 'roc', 'performance']
    elif style == 'regression': 
        metrics = ['taylor']
    
    for ax, metric in zip(axes.flat, metrics):

        xp = {}
        yp = {} 
        scores = {}
        pred = {}
        for name, model in estimators:
            if style == 'classification':
                predictions = model.predict_proba(X)[:,1]
            else:
                predictions = model.predict(X)
                
            _x, _y, _scores = sklearn_curve_bootstrap(
                                    y, 
                                    predictions, 
                metric=metric,
                n_boot=n_boot)
    
            xp[name] = _x
            yp[name] = _y
            pred[name] = predictions
            scores[name] = _scores
       
        if baseline_estimators is not None:
            for name, bl_model in baseline_estimators:
                predictions = bl_model.predict(X_baseline.reshape(-1, 1))
                
                _x, _y, _scores = sklearn_curve_bootstrap(
                                    y, 
                                    predictions, 
                metric=metric,
                n_boot=n_boot)
    
                xp[name] = _x
                yp[name] = _y
                pred[name] = predictions
                scores[name] = _scores
    
        verify.plot(diagram=metric, x=xp, y=yp, ax=ax, scores=scores, pred=pred)
    
    axes.flat[-1].remove()
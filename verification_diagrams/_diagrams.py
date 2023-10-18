import shapely.geometry

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from descartes import PolygonPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import math

import warnings
#from shapely.errors import ShapelyDeprecationWarning
#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

# Internal package(s)
from ._curve_utils import _confidence_interval_to_polygon
from ._metrics import calc_csi, reliability_uncertainty

class VerificationDiagram:
    def __init__(self, y_true=None, y_pred=None):
    
        self._y_true = y_true
        self._y_pred = y_pred 
        
        mpl.rcParams["axes.titlepad"] = 15
        mpl.rcParams["xtick.labelsize"] = 15
        mpl.rcParams["ytick.labelsize"] = 15
        
    
    def _add_major_and_minor_ticks(self, ax):
        """Add minor and major tick marks"""
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5, prune="lower"))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5, prune="lower"))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    
    def _set_axis_limits(self, ax,):
        """Sets the axis limits"""
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
    
    def make_twin_ax(self, ax):
        """
        Create a twin axis on an existing axis with a shared x-axis
        """
        # align the twinx axis
        twin_ax = ax.twinx()

        # Turn twin_ax grid off.
        twin_ax.grid(False)

        # Set ax's patch invisible
        ax.patch.set_visible(False)
        # Set axtwin's patch visible and colorize it in grey
        twin_ax.patch.set_visible(True)

        # move ax in front
        ax.set_zorder(twin_ax.get_zorder() + 1)

        return twin_ax
    
    def set_log_yticks(self, ax):

        cols = ax.patches
        vals = [p.get_height() for p in cols] 

        def orderOfMagnitude(number):
            return math.floor(math.log(number, 10))
        
        om = orderOfMagnitude(np.max(vals))
        ticks = [10**i for i in range(om+3) ]
    
        ax.set_yticks(ticks)
    
        return ax 
    
    def _plot_inset_ax(self, ax, line_colors):
        """ Plot the histogram associated with the reliability diagram """
        ax.grid(False)   
        bins = np.arange(0,1.1,0.1)
        df = pd.DataFrame(self._y_pred) 
        data = df.melt()

        # plot melted dataframe in a single command
        sns.histplot(data, x='value', hue='variable', palette=line_colors,
             multiple='dodge', shrink=.7, bins=bins, ax=ax, alpha=0.1, legend=False)

        ax.set_yscale('log')
        ax.set_ylabel(r'Samples', fontsize=10)    
        ax = self.set_log_yticks(ax)

        #ax.set_zorder(15)  # default zorder is 0 for ax1 and ax2
        #ax.set_frame_on(False)  # prevents ax1 from hiding ax2
        
        return ax
    
    def _plot_reliability_uncertainty(self, ax, line_colors):
        d=0.05
        for i, (name, color) in enumerate(zip(self._y_pred.keys(), line_colors)):
            _, _, ef_low, ef_up = reliability_uncertainty(self._y_true, self._y_pred[name], n_iter=100)

            x = np.linspace(0.5+(d*i),1,len(ef_low))
    
            ax.errorbar(x,x,yerr=[ef_low, ef_up], capsize=2.5, 
                fmt="o", ms=0.1, color=color, alpha=0.6)
    
    '''
    Deprecated. But don't want to delete just yet. 
    def _plot_inset_ax(self, ax, pred, line_colors, inset_yticks = [1e1, 1e3] ): 
        """Plot the inset histogram for the attribute diagram."""
        import math
        def orderOfMagnitude(number):
            return math.floor(math.log(number, 10))
        
        mag = orderOfMagnitude(len(pred))
        # Check if the number is even. 
        if mag % 2 == 0:
            mag+=1 

        inset_yticks = [10**i for i in range(mag)]
        inset_ytick_labels = [f'{10**i:.0e}' for i in range(mag)]
    
        # Histogram inset
        small_ax = inset_axes(
            ax,
            width="50%",
            height="50%",
            bbox_to_anchor=(0.15, 0.58, 0.5, 0.4),
            bbox_transform=ax.transAxes,
            loc=2,
        )

        small_ax.set_yscale("log", nonpositive="clip")
        small_ax.set_xticks([0, 0.5, 1])
        #small_ax.set_yticks(inset_yticks)
        #small_ax.set_yticklabels(inset_ytick_labels)
        #small_ax.set_ylim([1e0, np.max(inset_yticks)])
        small_ax.set_xlim([0, 1])
        
        bins=np.arange(0, 1.1, 0.1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        for k, color in zip(pred.keys(), line_colors):
            p = pred[k]
            fcst_probs = np.round(p, 5)
            n, x = np.histogram(a=fcst_probs, bins=bins)
            n = np.ma.masked_where(n==0, n)
            small_ax.plot(bin_centers, n, color=color, linewidth=0.6)
        
        return small_ax 
    '''    
    
    def _make_reliability(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
        fontsize = diagram_kwargs.get('fontsize', 12)
        add_axis_labels = diagram_kwargs.get('add_axis_labels', True)
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot([0,1], [0,1], ls='dashed', color='k', alpha=0.7)
        if add_axis_labels:
            ax.set_xlabel('Mean Forecast Probability', fontsize=fontsize)
            ax.set_ylabel('Conditional Event Frequency', fontsize=fontsize)
        
        if self._y_pred is not None:
            self._right_ax = self.make_twin_ax(ax)
     
        return ax 
    
    
    def _make_roc(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
        fontsize = diagram_kwargs.get('fontsize', 12)
        add_axis_labels = diagram_kwargs.get('add_axis_labels', True)
        
        pss_contours = diagram_kwargs.get('pss_contours', True)
        cmap = diagram_kwargs.get('cmap', 'Blues')
        alpha = diagram_kwargs.get('alpha', 0.6)
        
        x=np.arange(0,1.1,0.1)
        if pss_contours:
            # Compute the Pierce Skill Score (PSS)
            pod,pofd=np.meshgrid(x,x)
            pss = pod-pofd
            contours = ax.contourf(pofd, pod, pss, levels=x, cmap=cmap, alpha=alpha)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig = ax.get_figure()
            fig.colorbar(contours, cax=cax, label='Pierce Skill Score (POD-POFD)')
            
        # Plot random classifier/no-skill line 
        ax.plot(x,x,linestyle="dashed", color="gray", linewidth=0.8)
        
        if add_axis_labels:
            ax.set_xlabel('Probability of False Detection (POFD)', fontsize=fontsize)
            ax.set_ylabel('Probability of Detection (POD)', fontsize=fontsize)
        
        return ax, contours 

    def _make_performance(self, ax, **diagram_kwargs):
        """
        Make a performance diagram (Roebber 2009). 
        """
        fontsize = diagram_kwargs.get('fontsize', 12)
        add_axis_labels = diagram_kwargs.get('add_axis_labels', True)
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        xx = np.linspace(0.001,1,100)
        yy = xx
        xx,yy = np.meshgrid(xx,xx)
        csi = 1 / (1/xx + 1/yy -1)
        
        cf = ax.contourf(xx,yy,csi, cmap='Blues', alpha=0.3, levels=np.arange(0,1.1,0.1))
        if add_axis_labels:
            ax.set_xlabel('Success Ratio (SR; 1-FAR)', fontsize=fontsize)
            ax.set_ylabel('Probability of Detection (POD)', fontsize=fontsize)
        biasLines = ax.contour(
                    xx,
                    yy,
                    yy/xx,
                    colors="k",
                    levels=[0.5, 1.0, 1.5, 2.0, 4.0],
                    linestyles="dashed",
                    linewidths=0.5,
                    alpha=0.9
                    )
        ax.clabel(biasLines, levels=[0.5, 1.0, 1.5, 2.0, 4.0], fontsize=6, inline=True, fmt="%1.1f")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig = ax.get_figure()
        cb = fig.colorbar(cf, cax=cax)

        # Set the fontsize of the colorbar label
        cb.set_label('Critical Success Index (CSI)', fontsize=fontsize-2)
        
        return ax, cf 
        
    def plot(self, diagram, x, y,
             add_dots=True, 
             scores=None, 
             add_max_marker=False,
             line_colors=None,
             diagram_kwargs={}, 
             plot_kwargs={}, ax=None, add_table=True, table_bbox=None, 
             table_fontsize=8, table_alpha=0.5, pred=None): 
        """
        Plot a performance, attribute, or ROC Diagram. 
        
        Parameters
        ---------------
            diagram : 'performance', 'roc', or 'reliability'
            
            x,y : 1-d array, 2-d array or dict 
                The X and Y coordinate values. When plotting multiple 
                curves, then X and Y should be a dictionary. 
                E.g., x = {'Model 1' : x1, 'Model 2' : x2}
                      y = {'Model 1' : y1, 'Model 2' : y2}
                      
                If x or y are 2-d array, it is assumed the first
                dimension is from bootstrapping and will be used 
                to create confidence intervals. 
                      
            add_dots : True/False
            
        """
        if scores is not None:
            # Determine if scores is a nested dict. i.e., multiple models
            if not any(isinstance(i,dict) for i in scores.values()):
                scores = {'Model' : scores}

        plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
        plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 1.5)
        
        line_colors = plot_kwargs.get('line_colors', ['r', 'b', 'g', 'k', 'gray', 'purple'])
        line_styles = plot_kwargs.get('line_styles', ['-'])
        
        matplot_kwargs = plot_kwargs.copy()

        if 'line_colors' in matplot_kwargs.keys():
            matplot_kwargs.pop('line_colors')
                
        if 'line_styles' in matplot_kwargs.keys():
            matplot_kwargs.pop('line_styles')
        
        if ax is None:
            f, ax = plt.subplots(dpi=300, figsize=(4,4))
        
        self._set_axis_limits(ax)
        self._add_major_and_minor_ticks(ax)
        
        contours=None
        for_performance_diagram=False
        if diagram == 'performance':
            for_performance_diagram=True
            ax, contours = self._make_performance(ax=ax, **diagram_kwargs)
        elif diagram == 'roc':
            ax, contours = self._make_roc(ax=ax, **diagram_kwargs)
        elif diagram == 'reliability':
            ax = self._make_reliability(ax=ax, **diagram_kwargs)
            if self._y_pred is not None:
                self._plot_inset_ax(self._right_ax, line_colors)
                
            if self._y_pred is not None and self._y_true is not None:
                self._plot_reliability_uncertainty(ax, line_colors)
            
        else:
            raise ValueError(f'{diagram} is not a valid choice!')
    
        if not isinstance(x, dict):
            x = {'Label' : x}
            y = {'Label' : y}
        
        keys = x.keys()
        
        if len(line_styles) == 1:
            line_styles = line_styles*len(keys)
        
        error_bars=False
        for line_label, color, ls in zip(keys, line_colors, line_styles):
            _x = x[line_label]
            _y = y[line_label]
            matplot_kwargs['color'] = color
            matplot_kwargs['ls'] = ls
            
            if _x.ndim == 2:
                if _x.shape[0]>1:
                    error_bars=True
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _x = np.nanmean(_x, axis=0)
                    _y = np.nanmean(_y, axis=0)
                    
            line_label = None if line_label == 'Label' else line_label
    
            ax.plot(_x, _y, label=line_label,**matplot_kwargs)

            if diagram in ['roc', 'performance'] and add_dots:
                # Add scatter points at particular intervals 
                ax.scatter(_x[::20], _y[::20], s=100, marker=".", **matplot_kwargs)
                thresh = np.linspace(0,1,200)[::20]

                for i,j,t in zip(_x[::20], _y[::20], thresh):
                    ax.annotate(f'{int(t*100)}', (i,j), fontsize=7, zorder=6)
            
            csi = None
            if diagram in ['roc', 'performance'] and add_max_marker:
                if diagram == 'roc':
                    highest_val = np.argmax(_x - _y)
                else:
                    csi = calc_csi(_x, _y)
                    highest_val = np.argmax(csi)
            
                #matplot_kwargs['s'] = matplot_kwargs.get('s', 65)
                #matplot_kwargs['marker'] = matplot_kwargs.get('marker', 'X')
                ax.scatter(
                        _x[highest_val],
                        _y[highest_val],
                        s=65,
                        marker='X',
                        **matplot_kwargs, 
                        )
        
        if error_bars:
            # Adds the 95% confidence interval.
            for line_label, color in zip(keys, line_colors):
                _x = x[line_label]
                _y = y[line_label]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    x_coords_bottom, x_coords_top = np.nanpercentile(_x, (2.5, 97.5), axis=0)
                    y_coords_bottom, y_coords_top = np.nanpercentile(_y, (2.5, 97.5), axis=0)
                
                xci, yci = _confidence_interval_to_polygon(
                    x_coords_bottom,
                    y_coords_bottom,
                    x_coords_top,
                    y_coords_top,
                    for_performance_diagram=for_performance_diagram,
                )   
            
                ax.fill_between(xci, yci, 
                 color=color, alpha=0.4, interpolate=False)
            

        
        # Add the table of metrics to the verification diagrams
        # The code will attempt to determine the best location. 
        column_translator = {"NAUDPC" : 'NAPD', 'AUPDC' : 'APD'}
        
        if scores is not None:
            table_data, rows, columns = to_table_data(scores)
            rows = [f' {r} ' for r in rows]
           
            columns = [column_translator.get(c,c) for c in columns]
        
            n_rows, n_cols= np.shape(table_data)
            n_rows = max(1, n_rows)
            n_cols = max(1, n_cols)
            
            d_cols = n_cols-1
            shift = d_cols*0.16

            if diagram == 'performance':
                if csi is None: 
                    max_csi = np.max(calc_csi(_x, _y))
                else:
                    max_csi = np.max(csi)
                        
                if max_csi >= 0.4:
                    # For performance diagram curves in the upper right hand 
                    # corner place the table in the lower left.
                    xpos = 0.40-shift
                    if xpos < 0.1 or d_cols==0:
                        xpos=0.275
         
                    
                    bbox=[xpos, 0.025, 0.16*n_cols, 0.075*n_rows]        
                else:
                    # If the curve is in the lower left hand side, 
                    # then place the table in the upper right. 
                    bbox=[0.75-shift, 0.70, 0.16*n_cols, 0.075*n_rows]    
                        
            else:
                # For the ROC and Reliability diagram, we can place 
                # the table in the lower right hand corner. 
                bbox=[0.85, 0.025, 0.16*n_cols, 0.075*n_rows] 
           
            if table_bbox is None:
                table_bbox = bbox
                    
            if add_table:        
                plot_table(ax, table_data,
                    row_labels=rows,
                    column_labels=columns,
                    col_colors= None,
                    row_colors = {name : c for name,c in zip(rows, line_colors)},
                    bbox=table_bbox,
                    colWidth=0.16,
                    fontsize=table_fontsize, alpha=table_alpha)
            
        return ax
            
def plot_table(ax, table_data, row_labels, column_labels, row_colors, col_colors, bbox,
        fontsize=3., extra=0.7, colWidth=0.16, alpha=0.2 ):
    """
    Adds a table with the scores for each model.
    """
    col_colors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
    the_table = ax.table(cellText=table_data,
               rowLabels=row_labels,
               colLabels=column_labels,
               colWidths = [colWidth]*len(column_labels),
               rowLoc='right',
               cellLoc = 'center' , 
               colColours=col_colors,
               alpha=alpha,
               zorder=5,
               bbox=bbox
                )
    the_table.auto_set_font_size(False)
    table_props = the_table.properties()
    table_cells = table_props['children']
    i=0; idx = 0
    for cell in table_cells: 
        # Set transparency
        cell.set_alpha(alpha)
        
        cell_txt = cell.get_text().get_text()

        if i % len(column_labels) == 0 and i > 0:
            idx += 1
        
        if is_number(cell_txt):
            cell.get_text().set_fontsize(fontsize + extra)
            cell.get_text().set_color(row_colors[row_labels[idx]]) 
            
        else:
            cell.get_text().set_fontsize(fontsize-extra)
            
        if cell_txt in column_labels:
            cell.get_text().set_color('k') 
            if len(cell.get_text().__dict__['_text']) > 4:
                cell.get_text().set_fontsize(fontsize-2)
        else:
            pass
        
        if cell_txt in row_labels:
            # Change the row labels to have the right color and 
            # make them right-aligned
            cell.get_text().set_color(row_colors[cell_txt]) 
            cell.get_text().set_fontsize(fontsize)
            cell.get_text().set_ha('right')
        
        i+=1
        
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(0.125)
        

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

    
def to_table_data(scores):
    """Convert scores to tabular data format"""
    model_names = scores.keys()
    table_data = []
    for k in model_names:
        scorer_names = scores[k].keys()
        rows=[]
        for name in scorer_names:
            rows.append(np.nanmean(scores[k][name]))
        table_data.append(rows)
    
    table_data = np.round(table_data, 2)
    
    return table_data, list(model_names), list(scorer_names)
    
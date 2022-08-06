import shapely.geometry

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
from descartes import PolygonPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, f1_score, brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy import interpolate

import warnings
#from shapely.errors import ShapelyDeprecationWarning
#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 


class VerificationDiagram:
    mpl.rcParams["axes.titlepad"] = 15
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    
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
    
    def _make_reliability(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot([0,1], [0,1], ls='dashed', color='k', alpha=0.7)
        ax.set_xlabel('Mean Forecast Probability')
        ax.set_ylabel('Conditional Event Frequency')
        
        return ax 
    
    
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
        
    
    def _make_roc(self, ax, **diagram_kwargs):
        """
        Make the Receiver Operating Characterisitc (ROC) Curve.
        """
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
        
        return ax, contours 

    def _make_performance(self, ax, **diagram_kwargs):
        """
        Make a performance diagram (Roebber 2009). 
        """
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        xx = np.linspace(0.001,1,100)
        yy = xx
        xx,yy = np.meshgrid(xx,xx)
        csi = 1 / (1/xx + 1/yy -1)
        cf = ax.contourf(xx,yy,csi, cmap='Blues', alpha=0.3, levels=np.arange(0,1.1,0.1))
        ax.set_xlabel('Success Ratio (SR; 1-FAR)')
        ax.set_ylabel('Probability of Detection (POD)')
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
        fig.colorbar(cf, cax=cax, label='Critical Success Index (CSI)')

        return ax, cf 
        
    def plot(self, diagram, x, y,
             pred=None, 
             add_dots=True, 
             scores=None, 
             add_high_marker=False,
             line_colors=None,
             diagram_kwargs={}, 
             plot_kwargs={}, ax=None): 
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
            
            add_table : True/False
        """
        plot_kwargs['color'] = plot_kwargs.get('color', 'r')
        plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
        plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 1.5)
        
        line_colors = ['r', 'b', 'g', 'k']
        
        if ax is None:
            mpl.pyplot.subplots
            f, ax = plt.subplots(dpi=600, figsize=(4,4))
        
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
            if pred is not None:
                self._plot_inset_ax(ax, pred, line_colors)
            
        else:
            raise ValueError(f'{diagram} is not a valid choice!')
    
        if not isinstance(x, dict):
            x = {'Label' : x}
            y = {'Label' : y}
        
        keys = x.keys()
        
        error_bars=False
        for line_label, color in zip(keys, line_colors):
            _x = x[line_label]
            _y = y[line_label]
            plot_kwargs['color'] = color
            
            if _x.ndim == 2:
                error_bars=True
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    _x = np.nanmean(_x, axis=0)
                    _y = np.nanmean(_y, axis=0)
                    
            line_label = None if line_label == 'Label' else line_label
    
            ax.plot(_x, _y, label=line_label,**plot_kwargs)

            if diagram in ['roc', 'performance'] and add_dots:
                # Add scatter points at particular intervals 
                ax.scatter(_x[::10], _y[::10], s=15, marker=".", **plot_kwargs)
            
            
            if add_high_marker:
                if diagram == 'roc':
                    highest_val = np.argmax(_x - _y)
                else:
                    highest_val = np.argmax(csi)
            
                ax.scatter(
                        _x[highest_val],
                        _y[highest_val],
                        s=65,
                        marker = "X", 
                        **plot_kwargs, 
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
                
                polygon_object = _confidence_interval_to_polygon(
                    x_coords_bottom,
                    y_coords_bottom,
                    x_coords_top,
                    y_coords_top,
                    for_performance_diagram=for_performance_diagram,
                )   
            
                polygon_colour = mpl.colors.to_rgba(color, 0.4)
            
                polygon_patch = PolygonPatch(
                    polygon_object, lw=0, ec=polygon_colour, fc=polygon_colour
                )   
            
                ax.add_patch(polygon_patch)
        
        if scores is not None:
            if diagram == 'performance':
                loc = 'upper center'
            elif diagram == 'reliability':
                loc = 'lower right'
            elif diagram == 'roc':
                loc = 'center right'
            
            
            table_data, rows, columns = to_table_data(scores)
            
            rows = [f' {r} ' for r in rows]
            
            add_table(ax, table_data,
                    row_labels=rows,
                    column_labels=columns,
                    col_colors= None,
                    row_colors = {name : c for name,c in zip(rows, line_colors)},
                    loc=loc,
                    colWidth=0.16,
                    fontsize=8)
            
            
def add_table(ax, table_data, row_labels, column_labels, row_colors, col_colors, loc='best',
        fontsize=3., extra=0.75, colWidth=0.16, ):
    """
    Adds a table
    """
    #[0.12]*3
    col_colors = plt.cm.BuPu(np.full(len(column_labels), 0.1))
    the_table = ax.table(cellText=table_data,
               rowLabels=row_labels,
               colLabels=column_labels,
               colWidths = [colWidth]*len(column_labels),
               rowLoc='center',
               cellLoc = 'center' , 
               loc=loc, 
               colColours=col_colors,
               alpha=0.6,
               zorder=5
                )
    the_table.auto_set_font_size(False)
    table_props = the_table.properties()
    table_cells = table_props['children']
    i=0; idx = 0
    for cell in table_cells: 
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
            if len(cell.get_text().__dict__['_text']) > 3:
                cell.get_text().set_fontsize(fontsize-3.25)
        else:
            pass
            #cell.get_text().set_color('grey')
        
        if cell_txt in row_labels:
            cell.get_text().set_color(row_colors[cell_txt]) 
            cell.get_text().set_fontsize(fontsize)
        
        i+=1
        
    for key, cell in the_table.get_celld().items():
        cell.set_linewidth(0.25)

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
    
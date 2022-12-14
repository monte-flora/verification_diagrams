import numpy as np 
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, f1_score, brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy import interpolate
import shapely.geometry
from descartes import PolygonPatch

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

from ._metrics import reliability_curve, brier_skill_score

def bootstrap_generator(n_bootstrap, seed=42):
    """
    Create a repeatable bootstrap generator.
    """
    base_random_state = np.random.RandomState(seed)
    random_num_set = base_random_state.choice(10000, size=n_bootstrap, replace=False)
    return random_num_set

def sklearn_curve_bootstrap(y_true, y_pred, metric, n_boot=30, groups=None, scorers=None, 
                            random_seed=42, **kws):
    """Compute verification diagram curves within sklearn.metrics, but 
    with the ability to perform bootstrapping. For example, the sklearn ROC and 
    precisions-recall, and reliability/calibration curve are not consistently 
    shaped which prohibits easily bootstrapping. This code interpolates 
    the curves a fixed shape when bootstrapping. 
    
    Parameters
    ----------------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        True labels or binary label indicators. The binary and multiclass cases expect labels 
        with shape (n_samples,) while the multilabel case expects binary label indicators with shape (n_samples, n_classes).
        
    y_score : array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores.

    metric : 'performance', 'roc', or 'reliability'
        The verification diagram curve to compute for. 
    
    n_boot : int 
        Number of bootstrap iterations. 
        
    groups : array-like of shape (n_samples,) (default=None)
        array of indexs (0,1,2,etc) indicating groupings of the 
        predictions. This is useful when the data is not 
        fully independent and we want the bootstrapping 
        to be based on quasi-dependent samples. 
        The idea is that only data from a given index is 
        consider per bootstrap iteration with the index 
        being randomly chosen per bootstrap iteration. 
    
    scorers : dict of objects (default=None)
        For computational efficiency, you can provide a dict of 
        scorers to be computed per bootstrap iteration. 
        E.g., scorers = {'AUPDC' : average_precision_score, 
                         'AUC' : roc_auc_score}
    
    random_seed : int
        Random seed to control the randomness of the bootstrapping. 
    
    Returns
    -----------
       x : shape of (n_boot, 200) or (n_boot, 10)
           x-comp of the verification diagram curve
           
       y : shape of (n_boot, 200) or (n_boot, 200)
           y-comp of the verification diagram curve
       
       scores : dict of arrays of shape (n_boot,)
           The scores based on the scorer arg (see above)
    """
    if metric not in ['performance', 'roc', 'reliability']:
        raise ValueError(f"{metric} is not a valid option. Check for spelling errors.")
        
    N = 200
    if metric == 'performance':
        func = precision_recall_curve
        if scorers is None:
            scorers = {'AUPDC' : average_precision_score,}
    elif metric == 'roc':
        func = roc_curve
        if scorers is None:
            scorers = {'AUC' : roc_auc_score}
    elif metric == 'reliability':
        N = 10
        func = reliability_curve
        if scorers is None:
            scorers = {'BSS' : brier_skill_score} 
        
    curves = []
    scores = {k : [] for k in scorers.keys()}
    
    random_num_set = bootstrap_generator(n_boot, seed=random_seed)
    
    for i in range(n_boot):
        idx = resample(range(len(y_true)), replace=True, random_state=random_num_set[i])
        curves.append(func(y_true[idx], y_pred[idx], **kws))
        
        for k in scorers.keys():
            scores[k].append(scorers[k](y_true[idx], y_pred[idx]))
        
    sampled_thresholds = np.linspace(0.001, 0.99, N)
    sampled_x = []
    sampled_y = []
    # assume curves is a list of (precision, recall, threshold)
    # tuples where each of those three is a numpy array
    for pair in curves:
        if metric in ['performance', 'roc']:
            x, y, threshold = pair
            x_fp = x[:-1] if metric == 'performance' else x
            y_fp = y[:-1] if metric == 'performance' else y
            
            #x = np.interp(sampled_thresholds, threshold, x_fp)
            #y = np.interp(sampled_thresholds, threshold, y_fp)
            fx = interpolate.interp1d(threshold, x_fp, fill_value='extrapolate')
            fy = interpolate.interp1d(threshold, y_fp, fill_value='extrapolate')
            
            x = fx(sampled_thresholds)
            y = fy(sampled_thresholds)
        else:
            x, y = pair
        
        sampled_x.append(x)
        sampled_y.append(y)
    
    return np.array(sampled_x), np.array(sampled_y), scores


def _confidence_interval_to_polygon(
    x_coords_bottom,
    y_coords_bottom,
    x_coords_top,
    y_coords_top,
    for_performance_diagram=False,
):
    """Generates polygon for confidence interval.
    P = number of points in bottom curve = number of points in top curve
    :param x_coords_bottom: length-P np with x-coordinates of bottom curve
        (lower end of confidence interval).
    :param y_coords_bottom: Same but for y-coordinates.
    :param x_coords_top: length-P np with x-coordinates of top curve (upper
        end of confidence interval).
    :param y_coords_top: Same but for y-coordinates.
    :param for_performance_diagram: Boolean flag.  If True, confidence interval
        is for a performance diagram, which means that coordinates will be
        sorted in a slightly different way.
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    nan_flags_top = np.logical_or(np.isnan(x_coords_top), np.isnan(y_coords_top))
    if np.all(nan_flags_top):
        return None

    nan_flags_bottom = np.logical_or(
        np.isnan(x_coords_bottom), np.isnan(y_coords_bottom)
    )
    if np.all(nan_flags_bottom):
        return None

    real_indices_top = np.where(np.invert(nan_flags_top))[0]
    real_indices_bottom = np.where(np.invert(nan_flags_bottom))[0]

    if for_performance_diagram:
        y_coords_top = y_coords_top[real_indices_top]
        sort_indices_top = np.argsort(y_coords_top)
        y_coords_top = y_coords_top[sort_indices_top]
        x_coords_top = x_coords_top[real_indices_top][sort_indices_top]

        y_coords_bottom = y_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(-y_coords_bottom)
        y_coords_bottom = y_coords_bottom[sort_indices_bottom]
        x_coords_bottom = x_coords_bottom[real_indices_bottom][sort_indices_bottom]
    else:
        x_coords_top = x_coords_top[real_indices_top]
        sort_indices_top = np.argsort(-x_coords_top)
        x_coords_top = x_coords_top[sort_indices_top]
        y_coords_top = y_coords_top[real_indices_top][sort_indices_top]

        x_coords_bottom = x_coords_bottom[real_indices_bottom]
        sort_indices_bottom = np.argsort(x_coords_bottom)
        x_coords_bottom = x_coords_bottom[sort_indices_bottom]
        y_coords_bottom = y_coords_bottom[real_indices_bottom][sort_indices_bottom]

    polygon_x_coords = np.concatenate(
        (x_coords_top, x_coords_bottom, np.array([x_coords_top[0]]))
    )
    polygon_y_coords = np.concatenate(
        (y_coords_top, y_coords_bottom, np.array([y_coords_top[0]]))
    )

    return vertex_arrays_to_polygon_object(polygon_x_coords, polygon_y_coords)


def vertex_arrays_to_polygon_object(
    exterior_x_coords,
    exterior_y_coords,
    hole_x_coords_list=None,
    hole_y_coords_list=None,
):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.
    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole
    :param exterior_x_coords: np array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: np array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a np
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords
    )
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(
            _vertex_arrays_to_list(hole_x_coords_list[i], hole_y_coords_list[i])
        )

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords)
    )

    if not polygon_object.is_valid:
        raise ValueError("Resulting polygon is invalid.")

    return polygon_object


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.
    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).
    V = number of vertices
    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """
    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return np.array(vertex_coords_as_list)
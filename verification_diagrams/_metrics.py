from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, make_scorer
import numpy as np
import pandas as pd
from math import log, sqrt, cos, asin, log10, pi
import xarray as xr
from sklearn.utils import resample
from numpy.random import uniform

from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def brier_score(y, predictions):
    return np.mean((predictions - y) ** 2)

def brier_skill_score(y, predictions):
    return 1.0 - brier_score(y, predictions) / brier_score(y, y.mean())



def max_csi(y, predictions, known_skew):
    """
    Compute normalized maximum CSI 
    """
    sr, pod, _ = precision_recall_curve(y, predictions)
    sr[sr==0] = 0.0001
    pod[pod==0] = 0.0001
    
    csi = calc_csi(sr, pod)
    idx = np.argmax(csi)
    
    max_csi = csi[idx]
    norm_max_csi = norm_csi(y, predictions, known_skew)
    bias = pod / sr

    return {'MAX_CSI' : max_csi, 'NCSI' : norm_max_csi, 'FB' : bias[idx]}


def bss_reliability(y, predictions):
    """
    Reliability component of BSS. Weighted MSE of the mean forecast probabilities
    and the conditional event frequencies. 
    """
    mean_fcst_probs, event_frequency, indices = reliability_curve(y, predictions, n_bins=10, return_indices=True)
    # Add a zero for the origin (0,0) added to the mean_fcst_probs and event_frequency
    counts = [1e-5]
    for i in indices:
        if i is np.nan:
            counts.append(1e-5)
        else:
            counts.append(len(i[0]))

    mean_fcst_probs[np.isnan(mean_fcst_probs)] = 1e-5
    event_frequency[np.isnan(event_frequency)] = 1e-5

    diff = (mean_fcst_probs-event_frequency)**2
    return np.average(diff, weights=counts)


def modified_precision(precision, known_skew, new_skew): 
    """
    Modify the success ratio according to equation (3) from 
    Lampert and Gancarski (2014). 
    """
    precision[precision<1e-5] = 1e-5
    term1 = new_skew / (1.0-new_skew)
    term2 = ((1/precision) - 1.0)
    
    denom = known_skew + ((1-known_skew)*term1*term2)
    
    return known_skew / denom 
    
def calc_sr_min(skew):
    pod = np.linspace(0,1,100)
    sr_min = (skew*pod) / (1-skew+(skew*pod))
    return sr_min 

def _binary_uninterpolated_average_precision(
            y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if known_skew is not None:
            precision = modified_precision(precision, known_skew, new_skew)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def min_aupdc(y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None):
    """
    Compute the minimum possible area under the performance 
    diagram curve. Essentially, a vote of NO for all predictions. 
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    ap_min = _average_binary_score(average_precision, y_true, min_score,
                                 average, sample_weight=sample_weight)

    return ap_min


def calc_csi(precision, recall):
    """
    Compute the critical success index
    """
    precision[precision<1e-5] = 1e-3
    recall[recall<1e-5] = 1e-3
    
    csi = 1.0 / ((1/precision) + (1/recall) - 1.0)
    
    return csi 

def norm_csi(y_true, y_score, known_skew, pos_label=1, sample_weight=None):
    """
    Compute the normalized modified critical success index. 
    """
    new_skew = np.mean(y_true)
    precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if known_skew is not None:
        precision = modified_precision(precision, known_skew, new_skew)
    
    csi = calc_csi(precision, recall)
    max_csi = np.max(csi)
    ncsi = (max_csi - known_skew) / (1.0 - known_skew)
    
    return ncsi 
    
def norm_aupdc(y_true, y_score, known_skew, *, average="macro", pos_label=1,
                            sample_weight=None, min_method='random'):
    """
    Compute the normalized modified average precision. Normalization removes 
    the no-skill region either based on skew or random classifier performance. 
    Modification alters success ratio to be consistent with a known skew. 
  
    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1 
            Known or reference skew (# of 1 / n_samples) for 
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC 
            formula presented in Boyd et al. (2012).
            
            If 'random', then the normalization is based on the 
            minimum AUPDC for a random classifier, which is equal 
            to the known skew. 
    
    
    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    
    ap = _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
    
    if min_method == 'random':
        ap_min = known_skew 
    elif min_method == 'skew':
        ap_min = min_aupdc(y_true, 
                       pos_label, 
                       average,
                       sample_weight=sample_weight,
                       known_skew=known_skew, 
                       new_skew=new_skew)
    
    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc


# ESTABLISH THE SCORING METRICS FOR THE CROSS-VALIDATION
scorer_dict = {'auc': make_scorer(score_func=roc_auc_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    )    ,
           'aupdc': make_scorer(score_func=average_precision_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    )    ,
           'aupdc_norm': make_scorer(score_func=norm_aupdc,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    ),
           'bss' : make_scorer(score_func=brier_skill_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    ),
           }

class ContingencyTable:
    ''' Calculates the values of the contingency table.
    param: y, True binary labels. shape = [n_samples] 
    param: predictions, predictions binary labels. shape = [n_samples]
    ContingencyTable calculates the components of the contigency table, but ignoring correct negatives. 
    Can use determinstic and probabilistic input.     
    
    The 71+ metrics included here come from Brusco et al. (2021)
    "A comparison of 71 binary similarity coefficients: The effect of base rates"
    
    '''
    def __init__(self, y=None, predictions=None):
        
        if y is not None and predictions is not None:
            self._check_inputs(y, predictions)
            self._get_table_elements()
    
    
    def _get_table_elements(self):
        self.hits = np.sum((self.y == 1) & (self.predictions == 1))
        self.false_alarms = np.sum((self.y == 0) & (self.predictions == 1))
        self.misses = np.sum((self.y == 1) & (self.predictions == 0))
        self.corr_negs = np.sum((self.y == 0) & (self.predictions == 0))

        self.a = self.hits
        self.b = self.misses
        self.c = self.false_alarms
        self.d = self.corr_negs
        
        # Creating terms used in the equations below. 
        self.n = self.a+self.b+self.c+self.d
        
        self.tau_1 = max(self.a,self.b)+max(self.c,self.d)+max(self.a,self.c)+max(self.b,self.d)
        self.tau_2 = max(self.a+self.c, self.b+self.d) + max(self.a+self.b, self.c+self.d) 
        
        self._N = (self.n*(self.n-1))/2
        
        self._B = self.a*self.b + self.c*self.d
        self._C = self.a*self.c+self.b*self.d
        self._D = self.a*self.d+self.b*self.c
        
        self._A = self._N-self._B-self._C-self._D

    def _check_inputs(self, y, predictions):
        if isinstance(y, list):
            y = np.array(y)
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        if not self.is_binary(y):
            raise ValueError('y must be binary with values of 0 and 1')
            
        if not self.is_binary(predictions):
            raise ValueError('predictions must be binary with values of 0 and 1')
        
        if len(np.unique(y)) == 1:
            raise ValueError('y only has one class!')
            
        self.y = y
        self.predictions = predictions
        
    def check_denominator(self, denom):
        if denom < 0.0001:
            denom = 0.00001
        return denom
    
    def is_binary(self, arr):
        unique_vals = np.unique(arr)
        return np.array_equal(unique_vals, [0, 1]) or \
            np.array_equal(unique_vals, [1, 0]) or \
            np.array_equal(unique_vals, [0]) or \
            np.array_equal(unique_vals, [1])

    # Start of the metrics
    def pod(self, *args):
        return self.hits / (self.hits + self.misses)
    
    def pofd(self,*args):
        return self.false_alarms / (self.false_alarms + self.corr_negs)
    
    def sr(self,*args):
        return self.hits / (self.hits + self.false_alarms) if self.hits + self.false_alarms != 0 else 1.
    
    def dice_I(self,*args):
        return self.a / (self.a + self.b)
    
    def dice_II(self,*args):
        return self.a/(self.a+self.c)
    
    def jaccard(self,*args):
        return self.a/(self.a+self.b+self.c)
    
    def swjaccard(self,*args):
        return 3*self.a/(3*self.a+self.b+self.c)
    
    def gleason(self,*args):
        return 2*self.a/(2*self.a+self.b+self.c)
    
    def kulczynski_I(self,*args):
        return self.a/(self.b+self.c)
    
    def kulczynski_II(self,*args):
        return 0.5*(self.a/(self.a+self.b) + self.a/(self.a+self.c))
    
    def dko(self,*args):
        # Driver and Kroeber and Ochiai
        return self.a / (sqrt((self.a+self.b)*(self.a+self.c)))
    
    def braun_blanquet(self,*args):
        return self.a / max(self.a+self.b, self.a+self.c)
    
    def simpson(self,*args):
        return self.a / min(self.a+self.b, self.a+self.c)
    
    def sorgenfrei(self,*args):
        return self.a**2 / ((self.a+self.b)*(self.a+self.c))
    
    def mountford(self,*args):
        return 2*self.a / (self.a*self.b+ self.a*self.c + 2*self.b+self.c)
    
    def fager_and_mcgowan(self,*args):
        return self.dko() - max(self.a+self.b, self.a+self.c)/2
    
    def sokal_and_sneath_I(self,*args):
        return self.a / (self.a+2*self.b+2*self.c)
    
    def mcconaughey(self,*args):
        return (self.a**2-self.b*self.c)/(self.a+self.b*(self.a+self.c))
    
    def johnson(self,*args):
        return self.a/(self.a+self.b + self.a/(self.a+self.c))
    
    def van_der_maarel(self,*args):
        return (2*self.a-self.b+self.c)/(2*self.a+self.b+self.c)
    
    def ct_IV(self,*args):
        # Consonni and Todeschini
        return log(1+self.a)/log(1+self.a+self.b+self.c)
    
    def russel_and_rao(self,*args):
        return self.a / self.n 
    
    def ct_III(self,*args):
        # Consonni and Todeschini
        return log(1+self.a) / log(1+self.n)
    
    def sokal_and_michener(self,*args):
        return (self.a+self.d)/ self.n
    
    def rogers_and_tanimoto(self,*args):
        return (self.a+self.d) / (self.n+self.b+self.c)
    
    def sokal_and_sneath_II(self,*args):
        return (2*(self.a+self.d)) / (self.n+self.a+self.d)
    
    def sokal_and_sneath_III(self,*args):
        # 24 
        return (self.a+self.d) / (self.b+self.c)
    
    def faith(self,*args):
        return (self.a+(self.d/2)) / self.n
    
    def gower_and_legendre(self,*args):
        return (self.a+self.d) / (self.a+self.d+((self.b+self.c)/2))
    
    def gower(self,*args):
        return (self.a+self.d) / sqrt((self.a+self.b*(self.a+self.c)*(self.b+self.d)*(self.c+self.d)))
    
    def austin_and_colwell(self,*args):
        return (2/pi)*asin(sqrt((self.a+self.d)/self.n))
    
    def ct_I(self,*args):
        return log(1+self.a+self.d)/log(1+self.n)
    
    def hamann(self,*args):
        return (self.a+self.d-self.b-self.c) / (self.n)
    
    def peirce_I(self,*args):
        return (self.a*self.d - self.b*self.c)/ (self.a+self.b)*(self.c+self.d)
    
    def peirce_II(self,*args):
        return (self.a*self.d - self.b*self.c)/ (self.a+self.c)*(self.b+self.d)
    
    def yule_Q(self,*args):
        #33
        return (self.a*self.d-self.b*self.c)/ (self.a*self.d+self.b*self.c)
    
    def yule_W(self,*args):
        num = (sqrt(self.a*self.d)-sqrt(self.b*self.c))
        den = (sqrt(self.a*self.d)+sqrt(self.b*self.c))
    
        den = self.check_denominator(den)
        
        return num / den
    
    def pearson_I(self,*args):
        num = self.n*(self.a*self.d-self.b*self.c)**2 
        den = (self.a +self.d*self.a+self.c*self.b+self.d*self.c+self.d)
        
        den = self.check_denominator(den)
        
        return num / den
    
    def pearson_II(self,*args):
        chi_squared = self.pearson_I()
        return sqrt(chi_squared/ (self.n + chi_squared))
    
    def phi(self,*args):
        return (self.a*self.d-self.b*self.c)/ (sqrt(self.a+self.d*self.a+self.c*self.b+self.d*self.c+self.d))
    
    def michael(self,*args):
        return (4*(self.a *self.d-self.b*self.c)) / ((self.a +self.d**2 + (self.b +self.c**2)))
    
    def cole_I(self,*args):
        num = (self.a *self.d-self.b*self.c) 
        den = ((self.a +self.c) * (self.c +self.d))
        den = self.check_denominator(den)
        
        return num / den
    
    def cole_II(self,*args):
        return (self.a *self.d - self.b*self.c) / ((self.a + self.b) * (self.b +self.d))
    
    def cohen(self,*args):
        num =  (2*(self.a *self.d-self.b*self.c))
        den =  sqrt((self.a +self.b)*(self.b +self.d) + (self.a +self.c*(self.c +self.d)))
        
        den = self.check_denominator(den)
        
        return num/den     
    
    def maxwell_and_pilliner(self,*args):
        num = 2*(self.a*self.d-self.b*self.c) 
        den = ((self.a + self.b)*(self.c +self.d) + (self.a +self.c)*(self.b +self.d)) 
        
        den = self.check_denominator(den)
        
        return num/den
    
    def dennis(self,*args):
        num =(self.a*self.d-self.b*self.c) 
        den = sqrt(self.n*(self.a +self.b)*(self.a +self.c))
        den = self.check_denominator(den)
        
        return num/den
    
    def dispersion(self,*args):
        return (self.a *self.d-self.b*self.c) / self.n**2
    
    def ct_IV(self,*args):
        return (log(1 +self.a*self.d) - log(1 +self.b*self.c)) / log(1 + 0.25*self.n**2)
    
    def stiles(self,*args):
        num = self.n*(abs(self.a *self.d-self.b*self.c)-0.5*self.n)**2
        denom = (self.a+self.b)*(self.a+self.c)*(self.b+self.d)*(self.c+self.d)
        
        term = num/denom
        
        term = self.check_denominator(term)
        
        return log10(term)
    
    def scott(self,*args):
        num = 4 *self.a*self.d- (self.b +self.c)**2
        denom = (2 *self.a+self.b+self.c)*(2 *self.d+self.b+self.c)
        
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def tetrachoric(self,*args):
        term = (self.b *self.c)
        term = self.check_denominator(term)
        
        return cos(180./(1 + sqrt((self.a *self.d) / (term))))
    
    def odds_ratio(self,*args):
        num = (self.a *self.d)
        den = (self.b *self.c)
        
        den = self.check_denominator(den)
        
        return num/den
    
    def rand_coef(self,*args):
        return (self._A + self._B) / self._N
    
    def ari(self,*args):
        num1 =  self._N*(self._A + self._D) 
        num2 = (self._A + self._B)*(self._A + self._C)*(self._C + self._D)*(self._B + self._D)
        num = num1- num2
        denom = self._N**2 - (num2)
        
        denom = self.check_denominator(denom)
        
        return num / denom
    
    #def loevingers_h(self,*args):
    #    pass
    
    def sokal_and_sneath_V(self,*args):
        return 0.25*(self.dice_I() + self.dice_II() + self.d/(self.b+self.d) + self.d/(self.c+self.d))
    
    def sokal_and_sneath_V(self,*args):
        return (self.a *self.d) / sqrt((self.a +self.b*(self.a +self.c) + (self.b+self.d)*(self.c+self.d)))
    
    def rogot_and_goldberg(self,*args):
        term1 =self.a/ (2 *self.a+self.b+self.c)
        term2 =self.d/ (2 *self.d+self.b+self.c)
        return term1 + term2
    
    def baroni_and_buser_I(self,*args):
        return (sqrt(self.a *self.d) +self.a) / (sqrt(self.a *self.d +self.a+self.b+self.c))
    
    def peirce_III(self,*args):
        return (self.a *self.b+self.b*self.c)/ (self.a *self.b+ 2 *self.b*self.c+self.c*self.d)
    
    def hawkins_and_dotson(self,*args):
        term1 =self.a/ (self.a +self.b+self.c)
        term2 =self.d/ (self.b +self.c+self.d)
        return 0.5*(term1 + term2)
    
    def tarantula(self,*args):
        num =self.a* (self.c +self.d)
        denom =self.c* (self.a +self.b)
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def harris_and_lahey(self,*args):
        term1 =self.a* (2 *self.d+self.b+self.c) / (2 * (self.a +self.b+self.c))
        term2 =self.d* (2 *self.a+self.b+self.c) / (2 * (self.b +self.c+self.d))
        return term1+term2
    
    def forbes_I(self,*args):
        return self.n * self.a / ((self.a +self.b) * (self.a +self.c))
    
    def baroni_and_buser_II(self,*args):
        term = self.a *self.d +self.a-self.b-self.c
        if term <= 0:
            # Undefined so return 0 score. 
            return 0 
        
        return (sqrt(self.a *self.d +self.a-self.b-self.c)) / (sqrt(self.a *self.d +self.a+self.b+self.c))
    
    def fossum(self,*args):
        num = self.n * (self.a - 0.5)**2
        denom = sqrt((self.a +self.b * (self.a +self.c)))
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def forbes_II(self,*args):
        num = (self.n *self.a- (self.a +self.b) * (self.a +self.c))
        denom = self.n * ( min(self.a +self.b, self.a+self.c) - (self.a +self.b)* (self.a +self.c))
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def eyraud(self,*args):
        num = self.n**2 * ((self.n *self.a) - (self.a +self.b) * (self.a +self.c))
        denom = (self.a +self.b) * (self.a +self.c) * (self.b +self.d) * (self.c +self.d)
        
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def tarwid(self,*args):
        num   = self.n *self.a- ((self.a +self.b) * (self.a +self.c))
        denom = self.n *self.a+ ((self.a +self.b) * (self.a +self.c)) 
        
        denom = self.check_denominator(denom)
        
        return num/denom
    
    def goodman_and_kruskal_I(self,*args):
        return self.tau_1 - self.tau_2 / (2 * self.n - self.tau_2)
    
    def anderberg(self,*args):
        return (self.tau_1 - self.tau_2) / (2 *self.n)
    
    def goodman_and_kruskal_II(self,*args):
        num  = 2 * min(self.a ,self.d) -self.b- self.c
        den  = 2 * min(self.a ,self.d) +self.b+ self.c
        
        den = self.check_denominator(den)
        
        return num/den
    
    def gilbert_and_wells(self,*args):
        a = self.a 
        if a < 0.00001:
            a = 0.00001
            
        term =   (self.a+self.c)/ self.n
        if term < 0.000001:
            term = 0.000001 
            
        return log(a) - log(self.n) - log((self.a + self.b) / self.n) - log(term)
    
    def ct_II(self,*args):
        return (log(1+self.n) - log(1+self.b+self.c))/log(1+self.n)
                    
                    
def performance_curve(y, predictions, bins=np.arange(0, 1.025, 0.025), deterministic=False ):
    ''' 
    Generates the POD and SR for a series of probability thresholds 
    to produce performance diagram (Roebber 2009) curves
    '''
    predictions = np.round(predictions,5)
    
    if deterministic:
        table = ContingencyTable(y, predictions)
        pod = table.calc_pod( )
        sr = table.calc_sr( )
    else:
        tables = [ContingencyTable(y, np.where(predictions >= p, 1, 0)) for p in bins]

        pod = np.array([t.pod() for t in tables])
        sr = np.array([t.sr() for t in tables])
        
    return sr, pod

def roc_curve(y, predictions, bins=np.arange(0, 1.025, 0.025), deterministic=False ):
    ''' 
    Generates the POD and POFD for a series of probability thresholds 
    to produce the ROC curve. 
    '''    
    predictions = np.round(predictions,5)
    if deterministic:
        table = ContingencyTable(y, predictions)
        pod = table.pod( )
        sr = table.sr( )
    else:
        tables = [ContingencyTable(y, np.where(predictions >= p, 1, 0)) for p in bins]

        pod = np.array([t.pod() for t in tables])
        pofd = np.array([t.pofd() for t in tables])

    return pofd, pod 


#ChatGPT generated code. 
def reliability_curve(y_true, y_pred, n_bins=10, return_indices=False):
    """
    Generate a reliability (calibration) curve. 
    """
    bin_edges = np.linspace(0,1, n_bins+1)
    bin_indices = np.clip(np.digitize(y_pred, bin_edges, right=True) - 1, 0, None)

    mean_fcst_probs, event_frequency = [], []
    indices = []
    for i in range(n_bins):
        idx = np.where(bin_indices==i+1)
        mean_fcst_probs.append(np.mean(y_pred[idx]) if len(idx[0]) > 0 else np.nan)
        event_frequency.append(np.sum(y_true[idx]) / len(idx[0]) if len(idx[0]) > 0 else np.nan)
        indices.append(idx)

    # Adding the origin to the data
    mean_fcst_probs.insert(0,0)
    event_frequency.insert(0,0)
    
    if return_indices:
        return np.array(mean_fcst_probs), np.array(event_frequency), indices 
    else:
        return np.array(mean_fcst_probs), np.array(event_frequency) 

    
def reliability_uncertainty(y_true, y_pred, n_iter = 1000, n_bins=10 ):
    '''
    Calculates the uncertainty of the event frequency based on Brocker and Smith (WAF, 2007)
    '''
    mean_fcst_probs, event_frequency = reliability_curve(y_true, y_pred, n_bins=n_bins)

    event_freq_err = [ ]
    for i in range( n_iter ):
        Z     = uniform( size = len(y_pred) )
        X_hat = resample( y_pred )
        Y_hat = np.where( Z < X_hat, 1, 0 )
        _, event_freq = reliability_curve(X_hat, Y_hat, n_bins=n_bins)
        event_freq_err.append(event_freq)

    ef_low = np.nanpercentile(event_freq_err, 2.5, axis=0)
    ef_up  = np.nanpercentile(event_freq_err, 97.5, axis=0)

    return mean_fcst_probs, event_frequency, ef_low, ef_up


def _get_binary_xentropy(target_values, forecast_probabilities):
    """Computes binary cross-entropy.

    This function satisfies the requirements for `cost_function` in the input to
    `run_permutation_test`.

    E = number of examples

    :param: target_values: length-E numpy array of target values (integer class
        labels).
    :param: forecast_probabilities: length-E numpy array with predicted
        probabilities of positive class (target value = 1).
    :return: cross_entropy: Cross-entropy.
    """
    MIN_PROBABILITY = 1e-15
    MAX_PROBABILITY = 1. - MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities < MIN_PROBABILITY] = MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities > MAX_PROBABILITY] = MAX_PROBABILITY

    return -1 * np.nanmean(
        target_values * np.log2(forecast_probabilities) +
        (1 - target_values) * np.log2(1 - forecast_probabilities))


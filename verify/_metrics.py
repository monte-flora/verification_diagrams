

def brier_skill_score(y_values, forecast_probabilities, **kwargs):
    """Computes the brier skill score"""
    climo = np.mean((y_values - np.mean(y_values)) ** 2)
    return 1.0 - brier_score_loss(y_values, forecast_probabilities) / climo



def reliability_curve(targets, predictions, n_bins=10):
    """
    Generate a reliability (calibration) curve. 
    Bins can be empty for both the mean forecast probabilities 
    and event frequencies and will be replaced with nan values. 
    Unlike the scikit-learn method, this will make sure the output
    shape is consistent with the requested bin count. The output shape
    is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
    looks correct. 
    """
    bin_edges = np.linspace(0,1, n_bins+1)
    bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

    indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]

    mean_fcst_probs = [np.nan if i is np.nan else np.nanmean(predictions[i]) for i in indices]
    event_frequency = [np.nan if i is np.nan else np.sum(targets[i]) / len(i[0]) for i in indices]

    # Adding the origin to the data
    mean_fcst_probs.insert(0,0)
    event_frequency.insert(0,0)
        
    return np.array(mean_fcst_probs), np.array(event_frequency), indices
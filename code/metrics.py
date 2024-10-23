import numpy as np
from sklearn.metrics import auc
from scipy import interpolate

def remove_trailing_zeros_and_ones(recall, precision):
    """
    Remove trailing zeros and ones from recall and precision arrays.

    This function cleans up the input `recall` and `precision` arrays by 
    removing elements from the end of both arrays where the values are 0 or 1. 
    This is done to avoid spurious values affecting calculations such as 
    interpolation or the computation of metrics like AP (Average Precision).

    Parameters
    ----------
    recall : array-like
        Array of recall values.
    precision : array-like
        Array of precision values.

    Returns
    -------
    precision_cleaned : np.ndarray
        Precision array with trailing zeros and ones removed.
    recall_cleaned : np.ndarray
        Recall array with trailing zeros and ones removed.
    """
    
    # Find the last index where precision and recall are not 0 or 1
    last_valid_index = len(precision) - 1
    for i in range(len(precision) - 1, -1, -1):
        if precision[i] != 0 and precision[i] != 1 and recall[i] != 0 and recall[i] != 1:
            last_valid_index = i
            break

    # Slice both arrays to remove trailing zeros and ones
    precision_cleaned = np.array(precision[:last_valid_index + 1])
    recall_cleaned = np.array(recall[:last_valid_index + 1])
    
    return precision_cleaned, recall_cleaned

def compute_F1(recall, precision, eps=1e-6):
    """
    Compute the F1 score curve and the maximum F1 score.

    The F1 score is the harmonic mean of precision and recall. This function 
    calculates the F1 score at each point in the precision-recall curve and 
    returns the maximum F1 score along with its corresponding index.

    Parameters
    ----------
    recall : array-like
        Array of recall values.
    precision : array-like
        Array of precision values.
    eps : float, optional
        Small epsilon value to avoid division by zero (default is 1e-6).

    Returns
    -------
    f1_curve : np.ndarray
        F1 score at each point along the precision-recall curve.
    max_f1 : float
        The maximum F1 score.
    max_f1_index : int
        The index of the maximum F1 score.
    """
    
    # Calculate F1 score at each point (harmonic mean of precision and recall)
    f1_curve = 2 * (precision * recall) / (precision + recall + eps)

    # Get the index of the maximum F1 score
    max_f1_index = np.argmax(f1_curve)

    # Maximum F1 score
    max_f1 = f1_curve[max_f1_index]
    
    return f1_curve, max_f1, max_f1_index
    
    
def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) by interpolating the precision-recall curve.

    This function computes the Average Precision (AP) by interpolating the 
    precision-recall curve at 101 evenly spaced recall points. The area under 
    the curve (AUC) of the interpolated precision-recall curve is used to compute AP.

    Parameters
    ----------
    recall : array-like
        Array of recall values.
    precision : array-like
        Array of precision values.

    Returns
    -------
    ap : float
        The computed Average Precision (AP).
    recall_101 : np.ndarray
        Interpolated recall values at 101 evenly spaced points.
    precision_101 : np.ndarray
        Interpolated precision values at 101 evenly spaced points.
    """
    
    # Remove trailing zeros and ones from precision and recall
    precision, recall = remove_trailing_zeros_and_ones(recall, precision)

    # Append sentinel values to the beginning and end
    mrec = np.concatenate(([1.0], recall, [0.0]))
    mpre = np.concatenate(([0.0], precision, [1.0]))
    
    # Create an array of 101 evenly spaced points between 0 and 1 for recall interpolation
    recall_interp_points = np.linspace(0, 1, 101)

    # Perform interpolation for precision and recall
    precision_interp = interpolate.interp1d(np.linspace(0, 1, len(mpre)), mpre, kind='linear')
    recall_interp = interpolate.interp1d(np.linspace(0, 1, len(mrec)), mrec, kind='linear')

    # Get interpolated values at 101 points
    precision_101 = precision_interp(recall_interp_points)
    recall_101 = recall_interp(recall_interp_points)
    
    # Sort recall and precision by recall values to ensure recall is monotonically increasing
    sorted_indices = np.argsort(recall_101)
    recall_101 = np.array(recall_101)[sorted_indices]
    precision_101 = np.array(precision_101)[sorted_indices]

    # Compute the area under the curve (AUC) for the interpolated precision-recall curve
    ap = auc(recall_101, precision_101)
    
    return ap, recall_101, precision_101
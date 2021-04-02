import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    t = [ 1 if prediction[i] == ground_truth[i] else 0 for i in range(len(prediction))]
    f = [ 0 if prediction[i] == ground_truth[i] else 1 for i in range(len(prediction))]
    tsum = np.sum(t)
    fsum = np.sum(f)
    
    return tsum/(tsum+fsum)

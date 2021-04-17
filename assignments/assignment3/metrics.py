import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tp=[ 1 if prediction[i] == 1 and ground_truth[i] == 1 else 0 for i in range(len(prediction))]
    fp=[ 1 if prediction[i] == 1 and ground_truth[i] == 0 else 0 for i in range(len(prediction))]
    fn=[ 1 if prediction[i] == 0 and ground_truth[i] == 1 else 0 for i in range(len(prediction))]
    tn=[ 1 if prediction[i] == 0 and ground_truth[i] == 0 else 0 for i in range(len(prediction))]
    
    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)
    fn_sum = np.sum(fn)
    tn_sum = np.sum(tn)
    
    
    precision = tp_sum/(tp_sum+fp_sum)
    recall = tp_sum/(tp_sum+fn_sum)
    accuracy = (tp_sum+tn_sum)/(fp_sum+fn_sum+tp_sum+tn_sum)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    t = [ 1 if prediction[i] == ground_truth[i] else 0 for i in range(len(prediction))]
    f = [ 0 if prediction[i] == ground_truth[i] else 1 for i in range(len(prediction))]
    tsum = np.sum(t)
    fsum = np.sum(f)
    
    return tsum/(tsum+fsum)

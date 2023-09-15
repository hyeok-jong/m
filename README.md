# mimport sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve

def binary_metrics(output, label, threshold):
    '''
    output : model's output and before sigmoid
    '''
    assert output.shape == label.shape and len(output.shape) == 1, f'output shape : {output.shape}, label shape : {label.shape}'
    # model output -> probability
    prob = output.sigmoid()
    # probability -> prediction
    pred = (prob >= threshold)*1
    confusion = sklearn.metrics.confusion_matrix(
        y_true = label, 
        y_pred = pred
        )
    tn, fp, fn, tp = confusion.ravel()
    
    # metrics.accuracy_score
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # metrics.precision_score
    precision = tp / (tp + fp)
    
    # metrics.recall_score
    recall = tp / (tp + fn)
    
    # metrics.f1_score
    f1 = 2*(precision *recall) / (precision + recall)
    
    return {
        'TN' : tn,
        'FP' : fp,
        'FN' : fn,
        'TP' : tp,
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1
    }


def Gmean_metrics(output, label, threshold = None):
    assert output.shape == label.shape and len(output.shape) == 1, f'output shape : {output.shape}, label shape : {label.shape}'
    fpr, tpr, thresholds = roc_curve(
            y_true = label, 
            y_score = output.sigmoid())
    auc = sklearn.metrics.auc(x = fpr, y = tpr)
    filtered_thresholds = [thr for thr in thresholds if 0 <= thr <= 1]
    gmeans = np.sqrt(tpr * (1-fpr))
    best_index = np.argmax(gmeans)
    
    if threshold:
        best_threshold = threshold
    else:
        best_threshold = filtered_thresholds[best_index]
    metrics_dict = binary_metrics(output, label, threshold = best_threshold)
    metrics_dict.update({'threshold' : best_threshold})
    metrics_dict.update({'auc' : auc})
    return metrics_dict







def multi_accuracy(output : torch.tensor, target : torch.tensor):
    '''
    batch = mini batch or full batch
    output : (batch, class) : dosent matter after softmax or not
    target : (batch, )
    '''
    output = torch.softmax(output, dim = 1)
    pred = torch.argmax(output, dim = 1)

    conf_matrix = sklearn.metrics.confusion_matrix(target, pred) # np.array
    accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()
    return {'accuracy' : accuracy, 'conf_matrix' : conf_matrix}

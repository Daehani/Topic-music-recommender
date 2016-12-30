import numpy as np
import pandas as pd


def PRF(actual, predicted):
    """
    Calculate precision, recall f-measure
    Input
    ---------
    actual : list, actual listening tracks
    predicted : list, predicted listening tracks
    Output
    -------
    PRF : tuple, precision, recall, f1
    """
    if not actual:
        return (0, 0, 0)

    num_hits = 0
    precision = 0
    recall = 0
    for p in predicted:
        if p in actual:
            num_hits += 1
    precision = num_hits / len(predicted)
    recall = num_hits / len(actual)

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return (precision, recall, f1)

def avgPRF(resultDataframe, testDataframe):
    """
    Evaluate all users using precision, recall f-measure
    Input
    ---------
    resultDataframe : pd_dataframe, all user's actual listening tracks
    testDataframe : pd_dataframe, all user's predicted listening tracks
    Output
    -------
    avgPRF : float
    """
    users = resultDataframe['u_id'].unique()
    predicted_group = resultDataframe.groupby(['u_id'])['track_id'].unique()
    actual_group = testDataframe.groupby(['u_id'])['track_id'].unique()
    
    actual_list = []
    predicted_list = []
    for user in users:
        actual = [i for i in actual_group[user] if str(i) != 'nan']
        predicted = predicted_group[user].tolist()
        
        actual_list.append(actual)
        predicted_list.append(predicted)
    
    return np.mean([PRF(a, p) for a, p in zip(actual_list, predicted_list)], axis=0)

def AP(actual, predicted, k=10):
    """
    Calulate average precision
    Input
    ---------
    actual : list, actual listening tracks
    predicted : list, predicted listening tracks
    Output
    -------
    ave_precision : float, average precision
    """
    if not actual:
        return 0
    
    if len(predicted)>k:
        predicted = predicted[:k]  
              
    num_hits = 0
    score = 0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    
    ave_precision = score / min(len(actual), k) 
    
    return ave_precision

def MAP(resultDataframe, testDataframe, k=10):
    """
    Evaluate all users using average precision
    Input
    ---------
    actual : list, actual listening tracks
    predicted : list, predicted listening tracks
    Output
    -------
    mean_ap : float
    """
    users = resultDataframe['u_id'].unique()
    predicted_group = resultDataframe.groupby(['u_id'])['track_id'].unique()
    actual_group = testDataframe.groupby(['u_id'])['track_id'].unique()
    
    actual_list = []
    predicted_list = []
    for user in users:
        actual = [i for i in actual_group[user] if str(i) != 'nan']
        predicted = predicted_group[user].tolist()
        
        actual_list.append(actual)
        predicted_list.append(predicted)
    
    return np.mean([AP(a, p, k) for a, p in zip(actual_list, predicted_list)], axis=0)
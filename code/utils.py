from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt

def sample_image(img):
    # Displays midline of each axis of a 3D image in grayscale
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, int(img.shape[2]/2)], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, int(img.shape[1]/2), :], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img[int(img.shape[0]/2), :, :], cmap='gray')
    plt.show()

def print_stats(
    preds, labels
):
    # Prints AUC/Acc/Sense/Spec/NPV/PPV/NPV given preds and labels
    binary_preds = (preds >= 0.5).astype(int)
    
    print('AUC:', round(roc_auc_score(labels, preds), 2))
    print('Acc:', round(accuracy_score(labels, binary_preds), 2))
    
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (fp + tn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    
    print('Sens:', round(sens, 2))
    print('Spec:', round(spec, 2))
    print('PPV:', round(ppv, 2))
    print('NPV:', round(npv, 2))

def calc_CI_95(v, n):
    error = np.sqrt((v * (1-v)) / n)
    return error * 1.96

def show_best_threshold_stats_col(df, y_true_col, y_pred_col):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for j in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(df), len(df))
        if len(np.unique(df[y_true_col])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(df.loc[indices, y_true_col], df.loc[indices, y_pred_col])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

    print(f'AUC: {round(roc_auc_score(df[y_true_col], df[y_pred_col]), 2)} ({round(confidence_lower,2)} - {round(confidence_upper,2)})', )
    fpr, tpr, threshold = metrics.roc_curve(df[y_true_col], df[y_pred_col])
    best_threshold = max(tpr-fpr)
    print('threshold:', best_threshold)

    cr_dict = classification_report(df[y_true_col], df[y_pred_col] > best_threshold, output_dict=True)\
    
    acc = cr_dict['accuracy']
    acc_CI = calc_CI_95(acc, len(df))
    print(f'acc: {round(acc, 2)} ({round(acc - acc_CI, 2)} - {round(acc + acc_CI, 2)}) ({round(acc*len(df))}/{len(df)})')

    sens = cr_dict['1.0']['recall']
    sens_CI = calc_CI_95(sens, len(df))
    print(f'sens: {round(sens, 2)} ({round(sens - sens_CI, 2)} - {round(sens + sens_CI, 2)}) ({round(sens*len(df))}/{len(df)})')

    spec = cr_dict['0.0']['recall']
    spec_CI = calc_CI_95(spec, len(df))
    print(f'spec: {round(spec, 2)} ({round(spec - spec_CI, 2)} - {round(spec + spec_CI, 2)}) ({round(spec*len(df))}/{len(df)})')

    ppv = cr_dict['1.0']['precision']
    ppv_CI = calc_CI_95(ppv, len(df))
    print(f'ppv: {round(ppv, 2)} ({round(ppv - ppv_CI, 2)} - {round(ppv + ppv_CI, 2)}) ({round(ppv*len(df))}/{len(df)})')

    npv = cr_dict['0.0']['recall']
    npv_CI = calc_CI_95(npv, len(df))
    print(f'npv: {round(npv, 2)} ({round(npv - npv_CI, 2)} - {round(npv + npv_CI, 2)}) ({round(npv*len(df))}/{len(df)})')

    print('---------------------------------------------------------')
    print()

def show_best_threshold_stats_all(df, nums = None, titles = None):
    cols = list(df.columns)

    for i in range(int(len(cols)/4)):

        if nums:
            if not i in nums:
                continue

        if titles:
            print(titles[nums.index(i)])
        else:
            print(cols[i*4])
        print()

        # Internal
        print('internal')

        n_bootstraps = 1000
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for j in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(df), len(df))
            if len(np.unique(df[cols[(i*4)+1]])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(df.loc[indices, cols[(i*4)+1]], df.loc[indices, cols[(i*4)]])
            bootstrapped_scores.append(score)
            # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

        print(f'AUC: {round(roc_auc_score(df[cols[(i*4)+1]], df[cols[(i*4)]]), 2)} ({round(confidence_lower,2)} - {round(confidence_upper,2)})', )
        fpr, tpr, threshold = metrics.roc_curve(df[cols[(i*4)+1]], df[cols[(i*4)]])
        best_threshold = max(tpr-fpr)
        print('threshold:', best_threshold)

        cr_dict = classification_report(df[cols[(i*4)+1]], df[cols[(i*4)]] > best_threshold, output_dict=True)\
        
        acc = cr_dict['accuracy']
        acc_CI = calc_CI_95(acc, len(df))
        print(f'acc: {round(acc, 2)} ({round(acc - acc_CI, 2)} - {round(acc + acc_CI, 2)}) ({round(acc*len(df))}/{len(df)})')

        sens = cr_dict['1.0']['recall']
        sens_CI = calc_CI_95(sens, len(df))
        print(f'sens: {round(sens, 2)} ({round(sens - sens_CI, 2)} - {round(sens + sens_CI, 2)}) ({round(sens*len(df))}/{len(df)})')

        spec = cr_dict['0.0']['recall']
        spec_CI = calc_CI_95(spec, len(df))
        print(f'spec: {round(spec, 2)} ({round(spec - spec_CI, 2)} - {round(spec + spec_CI, 2)}) ({round(spec*len(df))}/{len(df)})')

        ppv = cr_dict['1.0']['precision']
        ppv_CI = calc_CI_95(ppv, len(df))
        print(f'ppv: {round(ppv, 2)} ({round(ppv - ppv_CI, 2)} - {round(ppv + ppv_CI, 2)}) ({round(ppv*len(df))}/{len(df)})')

        npv = cr_dict['0.0']['recall']
        npv_CI = calc_CI_95(npv, len(df))
        print(f'npv: {round(npv, 2)} ({round(npv - npv_CI, 2)} - {round(npv + npv_CI, 2)}) ({round(npv*len(df))}/{len(df)})')

        print()

        print('external')

        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for j in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(df), len(df))
            if len(np.unique(df[cols[(i*4)+1]])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(df.loc[indices, cols[(i*4)+3]], df.loc[indices, cols[(i*4)+2]])
            bootstrapped_scores.append(score)
            # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]

        print(f'AUC: {round(roc_auc_score(df[cols[(i*4)+3]], df[cols[(i*4)+2]]), 2)} ({round(confidence_lower,2)} - {round(confidence_upper,2)})', )

        fpr, tpr, threshold = metrics.roc_curve(df[cols[(i*4)+1]], df[cols[(i*4)]])
        best_threshold = max(tpr-fpr)
        print('threshold:', best_threshold)

        fpr, tpr, threshold = metrics.roc_curve(df[cols[(i*4)+3]], df[cols[(i*4)+2]])
        best_threshold = max(tpr-fpr)
        print('threshold:', best_threshold)

        cr_dict = classification_report(df[cols[(i*4)+3]], df[cols[(i*4)+2]] > best_threshold, output_dict=True)

        acc = cr_dict['accuracy']
        acc_CI = calc_CI_95(acc, len(df))
        print(f'acc: {round(acc, 2)} ({round(acc - acc_CI, 2)} - {round(acc + acc_CI, 2)}) ({round(acc*len(df))}/{len(df)})')

        sens = cr_dict['1.0']['recall']
        sens_CI = calc_CI_95(sens, len(df))
        print(f'sens: {round(sens, 2)} ({round(sens - sens_CI, 2)} - {round(sens + sens_CI, 2)}) ({round(sens*len(df))}/{len(df)})')

        spec = cr_dict['0.0']['recall']
        spec_CI = calc_CI_95(spec, len(df))
        print(f'spec: {round(spec, 2)} ({round(spec - spec_CI, 2)} - {round(spec + spec_CI, 2)}) ({round(spec*len(df))}/{len(df)})')

        ppv = cr_dict['1.0']['precision']
        ppv_CI = calc_CI_95(ppv, len(df))
        print(f'ppv: {round(ppv, 2)} ({round(ppv - ppv_CI, 2)} - {round(ppv + ppv_CI, 2)}) ({round(ppv*len(df))}/{len(df)})')

        npv = cr_dict['0.0']['recall']
        npv_CI = calc_CI_95(npv, len(df))
        print(f'npv: {round(npv, 2)} ({round(npv - npv_CI, 2)} - {round(npv + npv_CI, 2)}) ({round(npv*len(df))}/{len(df)})')

        print('---------------------------------------------------------')
        print()


### Utilizing fast delong method from X. Sun and W. Wu

import pandas as pd
import numpy as np
import scipy.stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)
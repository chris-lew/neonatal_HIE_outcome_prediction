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
        print(f'spec: {round(spec, 2)} ({round(spec - spec_CI, 2)} - {round(spec + spec_CI, 2)}) ({round(sens*len(df))}/{len(df)})')

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
        print(f'spec: {round(spec, 2)} ({round(spec - spec_CI, 2)} - {round(spec + spec_CI, 2)}) ({round(sens*len(df))}/{len(df)})')

        ppv = cr_dict['1.0']['precision']
        ppv_CI = calc_CI_95(ppv, len(df))
        print(f'ppv: {round(ppv, 2)} ({round(ppv - ppv_CI, 2)} - {round(ppv + ppv_CI, 2)}) ({round(ppv*len(df))}/{len(df)})')

        npv = cr_dict['0.0']['recall']
        npv_CI = calc_CI_95(npv, len(df))
        print(f'npv: {round(npv, 2)} ({round(npv - npv_CI, 2)} - {round(npv + npv_CI, 2)}) ({round(npv*len(df))}/{len(df)})')

        print('---------------------------------------------------------')
        print()
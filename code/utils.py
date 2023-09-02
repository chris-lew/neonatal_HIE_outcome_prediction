from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
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
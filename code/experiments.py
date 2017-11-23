#!/usr/bin/env python

import numpy as np
import smote
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def cross_validation(X, y):
    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10)
    clf = DecisionTreeClassifier()

    tprs = []
    fprs = []


    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(tpr[1])
        fprs.append(fpr[1])

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    mean_tpr = np.mean(tprs)
    mean_fpr = np.mean(fprs)
    print mean_fpr, mean_tpr

    # plt.plot(mean_fpr, mean_tpr, 'bo',label=r'Mean ROC ',lw=2, alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    return mean_fpr, mean_tpr


if __name__ == "__main__":

    data = smote.dataRead("mammography.csv")
    print data.shape

    X = data[:, [1, 0]]
    y = data[:, [data.shape[1] - 1]]

    pTPRa = []
    pFPRa = []

    pTPRb = []
    pFPRb = []

    minority_overSample_percent = [100, 200, 300, 400, 500]
    majority_underSample_percent = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]
    # majority_underSample_percent = [100]
    fpr, tpr = cross_validation(X, y[:, 0])

    pFPRa.append(fpr)
    pTPRa.append(tpr)

    pFPRb.append(fpr)
    pTPRb.append(tpr)

    unique, counts = np.unique(data[:, [data.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    for i in majority_underSample_percent:
        print '#####################' + str(i) + '%########################'
        dataB = smote.underSMOTE(data, i)

        unique, counts = np.unique(dataB[:, [data.shape[1] - 1]], return_counts=True)
        freq = dict(zip(unique, counts))
        nPositive = freq[1.0]
        nNegative = freq[0.0]
        print 'UnderSampled ' + str(i) + ' %'
        print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

        print '------------------------------------------------------------'

        dataC = smote.smote(data, 400, 5)

        unique, counts = np.unique(dataC[:, [data.shape[1] - 1]], return_counts=True)
        freq = dict(zip(unique, counts))
        nPositive = freq[1.0]
        nNegative = freq[0.0]
        print 'SMOTED 400 %'
        print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

        dataC = smote.underSMOTE(dataC, i)

        unique, counts = np.unique(dataC[:, [data.shape[1] - 1]], return_counts=True)
        freq = dict(zip(unique, counts))
        nPositive = freq[1.0]
        nNegative = freq[0.0]
        print 'UnderSMOTED '+str(i)+' %'
        print '+ve Class: ', nPositive, ' -ve Class: ', nNegative
        print '------------------------------------------------------------'

        X = dataC[:, [1, 0]]
        y = dataC[:, [data.shape[1] - 1]]

        fpr, tpr = cross_validation(X,y[:,0])

        pFPRa.append(fpr)
        pTPRa.append(tpr)

        X = dataB[:, [1, 0]]
        y = dataB[:, [data.shape[1] - 1]]

        fpr, tpr = cross_validation(X, y[:, 0])

        pFPRb.append(fpr)
        pTPRb.append(tpr)

    pFPRa.append(1.0)
    pTPRa.append(1.0)
    pFPRb.append(1.0)
    pTPRb.append(1.0)

    plt.plot(pFPRa, pTPRa, 'b*', linestyle='-', label=r'SMOTE 400%')
    plt.plot(pFPRb, pTPRb, 'ro', linestyle='-', label=r'Under')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

        # scores = cross_val_score(clfUSmote, X, Y[:,0], cv=10)
        # # print scores.shape
        # # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # print("Under-sample rate: %d  Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean(), scores.std() * 2))

        # fpr, tpr, thresholds = metrics.roc_curve(Y[:,0], scores, pos_label=2)







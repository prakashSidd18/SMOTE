#!/usr/bin/env python

import numpy as np
import smote
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


if __name__ == "__main__":

    data = smote.dataRead("mammography.csv")
    print data.shape

    minority_overSample_percent = [50, 100, 200, 300, 400, 500]
    majority_underSample_percent = [5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000, 2000]

    # 10-fold cross-validation
    for i in majority_underSample_percent:
        dataC = smote.smote(data, 400, 5)
        dataC = smote.underSMOTE(dataC, i)

        X = dataC[:, [1, 0]]
        Y = dataC[:, [data.shape[1] - 1]]

        print X.shape
        print Y[:,0].shape

        unique, counts = np.unique(dataC[:, [data.shape[1] - 1]], return_counts=True)
        freq = dict(zip(unique, counts))
        nPositive = freq[1.0]
        nNegative = freq[0.0]
        print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

        clfUSmote = DecisionTreeClassifier()
        scores = cross_val_score(clfUSmote, X, Y[:,0], cv=10)
        # print scores.shape
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Under-sample rate: %d  Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean(), scores.std() * 2))

        # fpr, tpr, thresholds = metrics.roc_curve(Y[:,0], scores, pos_label=2)







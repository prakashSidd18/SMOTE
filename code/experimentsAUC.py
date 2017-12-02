#!/usr/bin/env python

import numpy as np
import smote
from scipy import interp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import time

def cross_validation(X, y, clf, thr):
    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10)


    tprs = []
    aucs=[]
    mean_fpr = np.linspace(0, 1, 100)


    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        # k = 0
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_auc, mean_fpr, mean_tpr



if __name__ == "__main__":
    datasetNames = ['mammography', 'sat', 'pimaindians']
    dataset = datasetNames[2]
    data = smote.dataRead(dataset+".csv")

    minority_overSample_percent = [100, 200, 300, 400, 500]
    majority_underSample_percent = [10, 100, 150, 200, 500, 1000]

    visSample = 3
    NBthresh = 0.5
    thresh = 1.0
    KNN = 5

    pca_true = False
    if pca_true:
        xData = data[:, :data.shape[1]-1]
        yData = data[:, [data.shape[1]-1]]

        pca = PCA(n_components=2)
        xData = pca.fit_transform(xData)

        data = np.empty((data.shape[0], 3), dtype=xData.dtype)
        data[: , :2] = xData
        data[:, [2]] = yData

    t0 = time.time()
    X = data[:, [1, 0]]
    y = data[:, [data.shape[1] - 1]]

    exp = np.empty((len(majority_underSample_percent), len(minority_overSample_percent)+1))


    priors = []

    exp[:, 0] = 0



    unique, counts = np.unique(data[:, [data.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]

    for i in range(len(majority_underSample_percent)):
        # print '#######################################################################'
        dataB = smote.underSMOTE(data, majority_underSample_percent[i])

        unique, counts = np.unique(dataB[:, [data.shape[1] - 1]], return_counts=True)
        freq = dict(zip(unique, counts))
        nPositive = freq[1.0]
        nNegative = freq[0.0]
        print 'UnderSampled by ' + str(majority_underSample_percent[i]) + ' %'

        # Decision Tree classifier on Under-sampled majority data
        X = dataB[:, [1, 0]]
        y = dataB[:, [data.shape[1] - 1]]

        clf = DecisionTreeClassifier()
        aucv,fpr,tpr = cross_validation(X, y[:, 0], clf, thresh)
        exp[i,0] = aucv
        print 'SMOTED by ',
        for j in range(len(minority_overSample_percent)):
            #Over-sampled minority data on undersampled Majority Data
            dataC = smote.smote(dataB, minority_overSample_percent[j], 5)

            X = dataC[:, [1, 0]]
            y = dataC[:, [data.shape[1] - 1]]

            unique, counts = np.unique(dataC[:, [dataC.shape[1] - 1]], return_counts=True)
            freq = dict(zip(unique, counts))
            nPositive = freq[1.0]
            nNegative = freq[0.0]
            print str(minority_overSample_percent[j])+' %, ',

            # Decision Tree classifier on SMOTE+Under-sampled data
            clf = DecisionTreeClassifier()
            aucv, fpr, tpr  = cross_validation(X,y[:,0], clf, thresh)
            exp[i, j+1] = aucv

    t1 = time.time()
    timeTaken = t1-t0

    print dataset
    print 'Time elapsed: ', timeTaken
    print np.mean(exp,axis=0)
    print "####################################################################"
    print exp

    '''Sample Output'''
    '''
             Undersample       100% 	 200%        300%	      400%		 500%
    SatImage [ 0.78659246  0.78810681  0.82591577  0.83178657  0.83804842  0.82834139]
    Mammo    [ 0.72201504  0.72897763  0.77764215  0.79193469  0.79187588  0.77518559]
    pima     [ 0.65814065  0.65670094  0.7014472   0.71445631  0.72072788  0.73268711]
    '''
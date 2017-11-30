#!/usr/bin/env python

import numpy as np
import smote
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


def cross_validation(X, y, clf, thr):
    # 10-fold cross-validation
    cv = StratifiedKFold(n_splits=10)


    tprs = []
    fprs = []

    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        k = 0
        while k < len(thresholds):
            if thresholds[k] <= thr:
                break
            k += 1
        if k >= len(thresholds):
            k = len(thresholds) - 1

        tprs.append(tpr[k])
        fprs.append(fpr[k])

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    mean_tpr = np.mean(tprs)
    mean_fpr = np.mean(fprs)
    # print mean_fpr, mean_tpr

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
    datasetNames = ['mammography', 'sat', 'pimaindians']
    dataset = datasetNames[2]
    data = smote.dataRead(dataset+".csv")
    print data.shape

    minority_overSample_percent = [100, 200, 300, 400, 500]
    # majority_underSample_percent = [10, 50, 100, 125, 150, 175, 200, 500, 1000]
    majority_underSample_percent = [10, 15, 25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 800, 1000,
                                    2000]

    visSample = 1
    NBthresh = 0.5
    thresh = 1.0
    KNN = 5
    # minority_overSample_percent = [400]
    # majority_underSample_percent = [100, 300]

    pca_true = False
    if pca_true:
        xData = data[:, :data.shape[1]-1]
        yData = data[:, [data.shape[1]-1]]

        pca = PCA(n_components=2)
        xData = pca.fit_transform(xData)
        # print xData.shape

        data = np.empty((data.shape[0], 3), dtype=xData.dtype)
        data[: , :2] = xData
        data[:, [2]] = yData

    X = data[:, [1, 0]]
    y = data[:, [data.shape[1] - 1]]

    expA = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))
    expB = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))
    expC = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))
    expD = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))
    expE = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))
    expF = np.empty((len(minority_overSample_percent), len(majority_underSample_percent)+2, 2))

    priors = []
    clf = DecisionTreeClassifier()
    fpr, tpr = cross_validation(X, y[:, 0], clf, thresh)

    expA[:, 0, :] = [fpr, tpr]
    expB[:, 0, :] = [fpr, tpr]

    clf = GaussianNB()
    fpr, tpr = cross_validation(X, y[:, 0], clf, NBthresh)
    expC[:, 0, :] = [fpr, tpr]

    clf = KNeighborsClassifier(n_neighbors=KNN)
    fpr, tpr = cross_validation(X, y[:, 0], clf, thresh)
    expD[:, 0, :] = [fpr, tpr]
    expE[:, 0, :] = [fpr, tpr]
    expF[:, 0, :] = [fpr, tpr]

    unique, counts = np.unique(data[:, [data.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    for j in range(len(minority_overSample_percent)):
        # if j == visSample:
        for i in range(len(majority_underSample_percent)):
            print '#######################################################################'
            dataB = smote.underSMOTE(data, majority_underSample_percent[i])

            unique, counts = np.unique(dataB[:, [data.shape[1] - 1]], return_counts=True)
            freq = dict(zip(unique, counts))
            nPositive = freq[1.0]
            nNegative = freq[0.0]
            print 'UnderSampled by ' + str(majority_underSample_percent[i]) + ' %'
            print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

            print '------------------------------------------------------------'

            # Decision Tree classifier on Plain Under-sampled data
            X = dataB[:, [1, 0]]
            y = dataB[:, [data.shape[1] - 1]]

            clf = DecisionTreeClassifier()
            fpr, tpr = cross_validation(X, y[:, 0], clf, thresh)
            expB[j, i+1, :] = [fpr, tpr]

            # Gaussian Naive Bayes classifier on Plain Under-sampled data
            clf = GaussianNB()
            fpr, tpr = cross_validation(X, y[:, 0], clf, NBthresh)
            print 'Gauss NB Class Priors [-ve class, +ve class]: ', clf.class_prior_
            priors.append(clf.class_prior_)
            expC[j, i + 1, :] = [fpr, tpr]

            # k-NN classifier on Plain Under-sampled data
            clf = KNeighborsClassifier(n_neighbors=KNN)
            fpr, tpr = cross_validation(X, y[:, 0], clf, thresh)
            expE[j, i + 1, :] = [fpr, tpr]

            dataC = smote.smote(data, minority_overSample_percent[j], 5)

            X = dataC[:, [1, 0]]
            y = dataC[:, [data.shape[1] - 1]]

            unique, counts = np.unique(dataC[:, [data.shape[1] - 1]], return_counts=True)
            freq = dict(zip(unique, counts))
            nPositive = freq[1.0]
            nNegative = freq[0.0]
            print 'SMOTED by '+str(minority_overSample_percent[j])+' %'
            print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

            dataC = smote.underSMOTE(dataC, majority_underSample_percent[i])

            unique, counts = np.unique(dataC[:, [data.shape[1] - 1]], return_counts=True)
            freq = dict(zip(unique, counts))
            nPositive = freq[1.0]
            nNegative = freq[0.0]
            print 'UnderSMOTED by '+str(majority_underSample_percent[i])+' %'
            print '+ve Class: ', nPositive, ' -ve Class: ', nNegative
            print '------------------------------------------------------------'

            X = dataC[:, [1, 0]]
            y = dataC[:, [data.shape[1] - 1]]

            # Decision Tree classifier on SMOTE+Under-sampled data
            clf = DecisionTreeClassifier()
            fpr, tpr = cross_validation(X,y[:,0], clf, thresh)
            expA[j, i+1, :] = [fpr, tpr]

            # k-NN classifier on SMOTE+Under-sampled data
            clf = KNeighborsClassifier(n_neighbors=KNN)
            fpr, tpr = cross_validation(X, y[:, 0], clf, thresh)
            expD[j, i+1, :] = [fpr, tpr]

            print '#######################################################################'

        expA[:, -1, :] = 1.0
        expB[:, -1, :] = 1.0
        expC[:, -1, :] = 1.0
        expD[:, -1, :] = 1.0
        expE[:, -1, :] = 1.0
        expF[:, -1, :] = 1.0
        visSample = j

        cHullArr1 = np.concatenate((expA[visSample, :, :], expB[visSample, :, :], expC[visSample, :, :]))
        hull1 = ConvexHull(cHullArr1)

        plt.plot(expA[visSample, :, 0], expA[visSample, :, 1], 'b+', linestyle='-', label=r'SMOTE '+str(minority_overSample_percent[visSample])+'%')
        plt.plot(expB[visSample, :, 0], expB[visSample, :, 1], 'ro', linestyle='-', label=r'Under')
        plt.plot(expC[visSample, :, 0], expC[visSample, :, 1], 'k^', linestyle='-', label=r'Naive Bayes')
        plt.plot(1.0 , 1.0, 'g*', linestyle='-.', label=r'Convex Hull')
        for simplex in hull1.simplices:
            plt.plot(cHullArr1[simplex, 0], cHullArr1[simplex, 1], 'g*', linestyle='-.')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve - Decision Tree on '+dataset+' dataset')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('figures/'+dataset+'/ROC_1_'+str(minority_overSample_percent[visSample])+'.png')
        plt.close()

        cHullArr2 = np.concatenate((expD[visSample, :, :], expE[visSample, :, :]))
        hull2 = ConvexHull(cHullArr2)

        plt.plot(expD[visSample, :, 0], expD[visSample, :, 1], 'b+', linestyle='-', label=r'SMOTE '+str(minority_overSample_percent[visSample])+'%')
        plt.plot(expE[visSample, :, 0], expD[visSample, :, 1], 'ro', linestyle='-', label=r'Under')
        plt.plot(1.0, 1.0, 'g*', linestyle='-.', label=r'Convex Hull')
        for simplex in hull2.simplices:
            plt.plot(cHullArr2[simplex, 0], cHullArr2[simplex, 1], 'g*', linestyle='-.')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve - '+str(KNN)+'-NN on ' + dataset+' dataset')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('figures/'+dataset+'/ROC_2_'+str(minority_overSample_percent[visSample])+'.png')
        plt.close()

        print '-----------------------------------'
        print 'Priors: ', priors

        # scores = cross_val_score(clfUSmote, X, Y[:,0], cv=10)
        # # print scores.shape
        # # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # print("Under-sample rate: %d  Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean(), scores.std() * 2))

        # fpr, tpr, thresholds = metrics.roc_curve(Y[:,0], scores, pos_label=2)







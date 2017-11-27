#!usr/bin/env python

from sys import *
from os.path import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def dataRead(dataset):
    data = np.genfromtxt(dataset, delimiter=',')
    return data


def plot(clf, X, Y, id):
    print "plotting"
    plot_step = 0.02
    n_classes = 2
    plot_colors = 'rb'
    labels = {"1: Class 1", "2: Class 2"}
    # print data[:, :6]


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    x_test = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(x_test)
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    for i, color in zip(range(n_classes), plot_colors):
      idx = np.where(Y == i)
      plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=20)

    if id is 'a':
        method = 'Original Data'
    elif id is 'b':
        method = 'Over-Sampling with Replacement'
    elif id is 'c':
        method = 'SMOTE'
    elif id is 'd':
        method = 'Under-sampling + SMOTE'

    plt.suptitle("Decision surface of a decision tree using paired features-"+method)
    # plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    # plt.show
    name = id+'.png'
    plt.savefig(name)
    plt.close()


def overReplacement(data, percent):
    print 'Over-sampling with Replacement'
    N = percent / 100
    # print data.shape

    unique, counts = np.unique(data[:, [data.shape[1]-1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]

    nindex = 0
    # print '+ve Class: ', nPositive, ' -ve Class: ', nNegative
    posData = np.zeros((nPositive, data.shape[1]))
    for i in range(data.shape[0]):
        if data[i, [data.shape[1]-1]] == 1.0:
            # print i
            posData[nindex, :] = data[i, :]
            nindex+=1

    # print posData.shape


    nnindex = 0
    sz = data.shape[0]
    while nnindex < (N-1) * posData.shape[0]:
        i = np.random.randint(posData.shape[0])

        temp = np.empty((sz+1, data.shape[1]), data.dtype)
        temp[:sz, :] = data
        temp[data.shape[0], :] = posData[i, :]
        nnindex+=1
        sz+=1
        data = temp

    return data

def populate(posData, N, i, nnarray):

    # print posData.shape
    synthetic = np.empty((N, posData.shape[1]), posData.dtype)
    newindex = 0
    while N != 0:
        idx = np.random.randint(nnarray.shape[1])
        for attr in range(posData.shape[1]):
            # print attr
            if attr < posData.shape[1]-1:
                dif = posData[nnarray[i, idx], attr] - posData[i, attr]
                # print 'dif:', dif
                gap = np.random.random(1)
                # print 'gap:', gap
                synthetic[newindex, attr] = posData[i, attr] + gap * dif
            else:
                synthetic[newindex, attr] = posData[i, attr]
        newindex+=1
        N-=1

    return synthetic

def smote(data, percent, k):
    print 'SMOTE'
    unique, counts = np.unique(data[:, [data.shape[1]-1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]

    if percent < 100:
        nPositive = int(percent*1.0 / 100 )* nPositive
        percent = 100

    N = (int)(percent/100)


    # print '+ve Class: ', nPositive, ' -ve Class: ', nNegative
    posData = np.zeros((nPositive, data.shape[1]))
    nindex = 0
    for i in range(data.shape[0]):
        if data[i, [data.shape[1]-1]] == 1.0:
            # print i
            posData[nindex, :] = data[i, :]
            nindex += 1

    # print posData.shape

    sz = data.shape[0]
    nnarray = np.empty((nPositive, k), int)
    for i in range(nPositive):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(posData[:, [1, 0]])
        distances, indices = nbrs.kneighbors(posData[:, [1, 0]])
        nnarray[i, :] = indices[i, 1:]

        newRow = populate(posData, N-1, i, nnarray)

        temp = np.empty((sz + N-1, data.shape[1]), data.dtype)
        temp[:sz, :] = data

        for j in range(N-1):
            temp[sz, :] = newRow[j, :]
            sz += 1
        data = temp

    return data

def underSMOTE(data, percent):
    print 'Under-sampling'
    unique, counts = np.unique(data[:, [data.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]

    # if percent < 100:
    #     nPositive = int((percent*1.0 / 100) * nPositive)
    #     percent = 100

    N = (percent*1.0 / 100)
    
    targetNegative = (int) (nPositive / N)

    if targetNegative >= nNegative:
        return data

    negData = np.zeros((nNegative, data.shape[1]))
    nindex = 0
    for i in range(data.shape[0]):
        if data[i, [data.shape[1] - 1]] == 0.0:
            # print i
            negData[nindex, :] = data[i, :]
            nindex += 1

    posData = np.zeros((nPositive, data.shape[1]))
    nindex = 0
    for i in range(data.shape[0]):
        if data[i, [data.shape[1] - 1]] == 1.0:
            # print i
            posData[nindex, :] = data[i, :]
            nindex += 1
        if nindex == nPositive:
            break

    np.random.shuffle(negData)

    dat = np.empty((targetNegative+nPositive, data.shape[1]), dtype = data.dtype)
    dat[:targetNegative+1, :] = negData[:targetNegative+1, :]
    dat[targetNegative:, :] = posData

    np.random.shuffle(dat)

    return dat


if __name__=="__main__":

    load_data = True
    pca_true = True

    if load_data:
        dataset = abspath('mammo.csv')
        data = np.genfromtxt(dataset, delimiter=',')
    else:
        dataset = abspath('mammography.csv')
        data = dataRead(dataset)
        sz = data.shape[0]
        newsz = int(np.floor(0.1 * sz))
        data = data[np.random.choice(data.shape[0], newsz, replace=False), :]

        np.savetxt("mammo.csv", data, fmt='%10.5f', delimiter=',')

    # print Y
    # Reduce dimensionality of dataset to 2 using PCA for better classification
    if pca_true:
        xData = data[:, :data.shape[1]-1]
        yData = data[:, [data.shape[1]-1]]

        pca = PCA(n_components=2)
        xData = pca.fit_transform(xData)
        # print xData.shape

        dataPCA = np.empty((data.shape[0], 3), dtype=xData.dtype)
        dataPCA[:, :2] = xData
        dataPCA[:, [2]] = yData

    # Normal Decision tree CLassification
    X = dataPCA[:, [1, 0]]
    Y = dataPCA[:, [dataPCA.shape[1]-1]]

    unique, counts = np.unique(dataPCA[:, [dataPCA.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    clfNormal = DecisionTreeClassifier()
    clfNormal = clfNormal.fit(X, Y)
    plot(clfNormal, X, Y, 'a')

    # Over-sampling with replacement
    dataA = overReplacement(dataPCA, 500)

    X = dataA[:, [1, 0]]
    Y = dataA[:, [dataA.shape[1]-1]]

    unique, counts = np.unique(dataA[:, [dataA.shape[1]-1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    clfReplace = DecisionTreeClassifier()
    clfReplace = clfReplace.fit(X, Y)
    plot(clfReplace, X, Y, 'b')

    # SMOTE
    dataB = smote(dataPCA, 500, 5)

    X = dataB[:, [1, 0]]
    Y = dataB[:, [dataB.shape[1]-1]]

    unique, counts = np.unique(dataB[:, [dataB.shape[1]-1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    clfSmote = DecisionTreeClassifier()
    clfSmote = clfReplace.fit(X, Y)
    plot(clfSmote, X, Y, 'c')


    # Under-sampling and SMOTE

    dataC = smote(dataPCA, 500, 5)
    dataC = underSMOTE(dataC, 100)

    X = dataC[:, [1, 0]]
    Y = dataC[:, [dataC.shape[1] - 1]]

    unique, counts = np.unique(dataC[:, [dataC.shape[1] - 1]], return_counts=True)
    freq = dict(zip(unique, counts))
    nPositive = freq[1.0]
    nNegative = freq[0.0]
    print '+ve Class: ', nPositive, ' -ve Class: ', nNegative

    clfUSmote = DecisionTreeClassifier()
    clfUSmote = clfUSmote.fit(X, Y)
    plot(clfUSmote, X, Y, 'd')









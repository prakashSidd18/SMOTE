#!usr/bin/env python

from sys import *
from os.path import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt


def dataRead(dataset):
    data = np.genfromtxt(dataset, delimiter=',')

    sz = data.shape[0]
    newsz = int(np.floor(0.1 * sz))
    data = data[np.random.choice(data.shape[0], newsz, replace=False), :]
    return data


def plot(clf, X, Y, id):
    print "plotting"
    plot_step = 0.02
    n_classes = 2
    plot_colors = 'gr'
    labels = {"1: Class 1", "2: Class 2"}
    # print data[:, :6]


    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    for i, color in zip(range(n_classes), plot_colors):
      idx = np.where(Y == i)
      plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    if id is 'a':
        method = 'Original Data'
    elif id is 'b':
        method = 'Over-Sampling with Replacement'
    elif id is 'c':
        method = 'SMOTE'

    plt.suptitle("Decision surface of a decision tree using paired features-"+method)
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    # plt.show
    name = id+'.png'
    plt.savefig(name)
    plt.close()


def overReplacement(data, percent):
    print 'Over-sampling with Replacement'
    N = percent / 100
    # print data.shape

    unique, counts = np.unique(data[:, [6]], return_counts=True)
    freq = dict(zip(unique, counts))
    nNeg = freq[1.0]
    nPos = freq[0.0]

    nindex = 0
    # print 'Neg: ', nNeg, ' Pos: ', nPos
    negData = np.zeros((nNeg, 7))
    for i in range(data.shape[0]):
        if data[i, [6]] == 1.0:
            # print i
            negData[nindex, :] = data[i, :]
            nindex+=1

    # print negData.shape


    nnindex = 0
    sz = data.shape[0]
    while nnindex < (N-1) * negData.shape[0]:
        i = np.random.randint(negData.shape[0])

        temp = np.empty((sz+1, data.shape[1]), data.dtype)
        temp[:sz, :] = data
        temp[data.shape[0], :] = negData[i, :]
        nnindex+=1
        sz+=1
        data = temp

    return data

def populate(negData, N, i, nnarray):

    # print negData.shape
    synthetic = np.empty((N, negData.shape[1]), negData.dtype)
    newindex = 0
    while N != 0:
        idx = np.random.randint(nnarray.shape[1])
        for attr in range(negData.shape[1]):
            # print attr
            if attr < 6:
                dif = negData[nnarray[i, idx], attr] - negData[i, attr]
                # print 'dif:', dif
                gap = np.random.random(1)
                # print 'gap:', gap
                synthetic[newindex, attr] = negData[i, attr] + gap * dif
            else:
                synthetic[newindex, attr] = negData[i, attr]
        newindex+=1
        N-=1

    return synthetic

def smote(data, percent, k):
    print 'SMOTE'
    unique, counts = np.unique(data[:, [6]], return_counts=True)
    freq = dict(zip(unique, counts))
    nNeg = freq[1.0]
    nPos = freq[0.0]

    if(percent < 100):
        nNeg = (percent/100)*nNeg
        N = 100

    N = (int)(percent/100)

    nindex = 0
    # print 'Neg: ', nNeg, ' Pos: ', nPos
    negData = np.zeros((nNeg, 7))
    for i in range(data.shape[0]):
        if data[i, [6]] == 1.0:
            # print i
            negData[nindex, :] = data[i, :]
            nindex += 1

    # print negData.shape

    sz = data.shape[0]
    nnarray = np.empty((nNeg, k), int)
    for i in range(nNeg):
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(negData[:, [1, 0]])
        distances, indices = nbrs.kneighbors(negData[:, [1, 0]])
        nnarray[i, :] = indices[i, 1:]
        newRow = populate(negData, N-1, i, nnarray)
        temp = np.empty((sz + 1, data.shape[1]), data.dtype)
        temp[:sz, :] = data
        temp[data.shape[0], :] = newRow
        sz += 1
        data = temp

    return data



if __name__=="__main__":

    load_data = True

    if load_data:
        dataset = abspath('mammo.csv')
        data = np.genfromtxt(dataset, delimiter=',')
    else:
        dataset = abspath('mammography.csv')
        data = dataRead(dataset)
        np.savetxt("mammo.csv", data, fmt='%10.5f', delimiter=',')

    # print Y

    # Normal Decision tree CLassification
    X = data[:, [1, 0]]
    Y = data[:, [6]]
    clfNormal = DecisionTreeClassifier()
    clfNormal = clfNormal.fit(X, Y)
    # plot(clfNormal, X, Y, 'a')

    # Over-sampling with replacement

    dataA = overReplacement(data, 200)
    np.random.shuffle(dataA)
    # print dataA.shape
    X = dataA[:, [1, 0]]
    Y = dataA[:, [6]]

    unique, counts = np.unique(dataA[:, [6]], return_counts=True)
    freq = dict(zip(unique, counts))
    nNeg = freq[1.0]
    nPos = freq[0.0]
    print 'Neg: ', nNeg, ' Pos: ', nPos

    clfReplace = DecisionTreeClassifier()
    clfReplace = clfReplace.fit(X, Y)
    # plot(clfReplace, X, Y, 'b')

    # SMOTE

    dataB = smote(data, 200, 5)
    np.random.shuffle(dataB)
    # print dataA.shape
    X = dataB[:, [1, 0]]
    Y = dataB[:, [6]]

    unique, counts = np.unique(dataB[:, [6]], return_counts=True)
    freq = dict(zip(unique, counts))
    nNeg = freq[1.0]
    nPos = freq[0.0]
    print 'Neg: ', nNeg, ' Pos: ', nPos

    clfSmote = DecisionTreeClassifier()
    clfSmote = clfReplace.fit(X, Y)
    # plot(clfSmote, X, Y, 'c')


    # Under-sampling and SMOTE









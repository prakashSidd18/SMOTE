#!usr/bin/env python

from sys import *
from os.path import *
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

dataset = abspath('mammography.csv')
data = np.genfromtxt(dataset, delimiter=',')
sz = data.shape[0]
newsz = int(np.floor(0.1*sz))

data = data[np.random.choice(data.shape[0], newsz, replace=False), :]

plot_step = 0.02
n_classes =  2
plot_colors = 'gr'
# print data[:, :6]

X = data[:, [1, 2]]
Y = data[:, [6]]

# print X
# print 'and'
# print Y

clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color,
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()




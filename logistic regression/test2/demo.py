# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from glm import LogisticRegression


# step1: classifier
clf = LogisticRegression(2)


# step2: fit
iris = datasets.load_iris()
train_x = iris.data[:100, :2]
train_y = iris.target[:100]
clf.fit(train_x, train_y)


# step3: predict
y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
ZZ = np.array(Z).reshape(xx.shape)


# step4: plot
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure()
plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light)
plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

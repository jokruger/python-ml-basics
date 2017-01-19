#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas, numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# read data
dataset = pandas.read_csv('data.csv')
array = dataset.values
X, Y = array[:, 0:-1], array[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=13)

# fit scaler
scaler = StandardScaler()
scaler.fit(X_train)

# scale data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# fit model
#model = SVC(kernel="linear", C=0.025)
#model = SVC(gamma=2, C=1)
model = KNeighborsClassifier(10)
#model = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
#model = DecisionTreeClassifier(max_depth=5)
#model = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)
#model = MLPClassifier(alpha=1)
#model = AdaBoostClassifier()
#model = GaussianNB()
#model = QuadraticDiscriminantAnalysis()

model.fit(X_train, Y_train)

# cals predictions
if hasattr(model, "decision_function"):
    Y_train_predicted = model.decision_function(X_train)
    Y_test_predicted = model.decision_function(X_test)
else:
    Y_train_predicted = model.predict_proba(X_train)[:, 1]
    Y_test_predicted = model.predict_proba(X_test)[:, 1]

fpr_train, tpr_train, _ = roc_curve(Y_train, Y_train_predicted)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(Y_test, Y_test_predicted)
roc_auc_test = auc(fpr_test, tpr_test)

natural_precision = precision_score(Y_test, [1]*len(Y_test))
natural_recall = recall_score(Y_test, [1]*len(Y_test))

threshold = numpy.linspace(0.01, 0.99, num=100)
precision = [precision_score(Y_test, [1 if j > i else 0 for j in Y_test_predicted]) for i in threshold]
recall = [recall_score(Y_test, [1 if j > i else 0 for j in Y_test_predicted]) for i in threshold]

# calc decision boundary
x = scaler.transform(X)
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
x1, x2 = numpy.meshgrid(numpy.linspace(x1_min, x1_max, 100), numpy.linspace(x2_min, x2_max, 100))
if hasattr(model, "decision_function"):
    y = model.decision_function(numpy.c_[x1.ravel(), x2.ravel()])
else:
    y = model.predict_proba(numpy.c_[x1.ravel(), x2.ravel()])[:, 1]
y = y.reshape(x1.shape)

# display charts
plt.figure(figsize=(12,10))

plt.subplot(221)
plt.plot(fpr_train, tpr_train, color='#8D8D8D', lw=2, label='ROC curve for train data (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='#F15854', lw=2, label='ROC curve for test data (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right", prop={'size': 8})

plt.subplot(222)
plt.plot(threshold, [natural_precision]*len(threshold), lw=2, color='#7CB1D7', linestyle='--', label='Natural precision')
plt.plot(threshold, [natural_recall]*len(threshold), lw=2, color='#FCBB6B', linestyle='--', label='Natural recall')
plt.plot(threshold, precision, lw=2, color='#5DA5DA', label='Precision curve')
plt.plot(threshold, recall, lw=2, color='#FAA43A', label='Recall curve')
plt.xlabel('Threshold')
plt.ylabel('Precision, Recall')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Precision, Recall')
plt.legend(loc="lower left", prop={'size': 8})

plt.subplot(223)
plt.scatter(x[:, 0], x[:, 1], c=Y, cmap=ListedColormap(['#FAA43A', '#5DA5DA']))
plt.xlim([x1_min, x1_max])
plt.ylim([x2_min, x2_max])
plt.xlabel('X2')
plt.ylabel('X1')
plt.title('Input data (scaled)')

plt.subplot(224)
plt.contourf(x1, x2, y, cmap=LinearSegmentedColormap.from_list('cm', ['#FAA43A', '#5DA5DA']), alpha=.8)
plt.xlim([x1_min, x1_max])
plt.ylim([x2_min, x2_max])
plt.xlabel('X2')
plt.ylabel('X1')
plt.title('Classification probability (class 0)')

plt.show()

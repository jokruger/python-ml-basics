#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas, numpy, sklearn
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model, tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score

dataset = pandas.read_csv('data.csv')

array = dataset.values
X = array[:, 0:-1]
Y = array[:, -1]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.5, random_state=13)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = linear_model.LinearRegression()
#model = linear_model.LogisticRegression(C=1, penalty='l1', tol=0.01)
#model = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.1)
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 1), random_state=1)
#model = RandomForestClassifier(n_estimators=500, n_jobs=3, oob_score=True)

model.fit(X_train, Y_train)

if 'predict_proba' in dir(model):
    if type(model) is RandomForestClassifier:
        Y_train_predicted = model.oob_decision_function_[:,1].tolist()
    else:
        Y_train_predicted = model.predict_proba(X_train)[:,1].tolist()

    Y_test_predicted = model.predict_proba(X_test)[:,1].tolist()
else:
    Y_train_predicted = model.predict(X_train)
    Y_test_predicted = model.predict(X_test)

fpr_train, tpr_train, _ = roc_curve(Y_train, Y_train_predicted)
roc_auc_train = auc(fpr_train, tpr_train)

fpr_test, tpr_test, _ = roc_curve(Y_test, Y_test_predicted)
roc_auc_test = auc(fpr_test, tpr_test)

natural_precision = precision_score(Y_test, [1]*len(Y_test))
natural_recall = recall_score(Y_test, [1]*len(Y_test))

threshold = numpy.linspace(0.01, 0.99, num=100)
precision = [precision_score(Y_test, [1 if j > i else 0 for j in Y_test_predicted]) for i in threshold]
recall = [recall_score(Y_test, [1 if j > i else 0 for j in Y_test_predicted]) for i in threshold]

# display charts
plt.figure(figsize=(16,6), dpi=97)

plt.subplot(121)
plt.plot(fpr_train, tpr_train, color='#8D8D8D', lw=2, label='ROC curve for train data (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='#F15854', lw=2, label='ROC curve for test data (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right", prop={'size': 8})

plt.subplot(122)
plt.plot(threshold, [natural_precision]*len(threshold), lw=2, color='#7CB1D7', linestyle='--', label='Natural precision')
plt.plot(threshold, [natural_recall]*len(threshold), lw=2, color='#FCBB6B', linestyle='--', label='Natural recall')
plt.plot(threshold, precision, lw=2, color='#5DA5DA', label='Precision curve')
plt.plot(threshold, recall, lw=2, color='#FAA43A', label='Recall curve')
plt.xlabel('Threshold')
plt.ylabel('Precision, Recall')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision, Recall')
plt.legend(loc="lower left", prop={'size': 8})

plt.show()

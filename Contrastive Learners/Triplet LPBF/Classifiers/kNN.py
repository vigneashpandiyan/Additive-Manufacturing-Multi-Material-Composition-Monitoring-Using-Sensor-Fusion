# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:04:37 2022

@author: srpv
"""
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn import neighbors, datasets
import os

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# %%


def KNN(X_train, X_test, y_train, y_test, classes, neigh, weights, total_path):

    print('Model to be trained is KNN')

    model = neighbors.KNeighborsClassifier(n_neighbors=neigh, weights=weights)
    model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    predictions = model.predict(X_test)
    print("KNN Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'kNN'+'_without normalization w/o Opt'
    graph_name2 = 'k-Nearest Neighbours'

    graph_1 = 'kNN'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'kNN'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Greys, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')
        plt.title(title, size=12)
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', dpi=400)
        plt.show()

    savemodel = 'KNN'+'_model'+'.sav'
    joblib.dump(model, savemodel)

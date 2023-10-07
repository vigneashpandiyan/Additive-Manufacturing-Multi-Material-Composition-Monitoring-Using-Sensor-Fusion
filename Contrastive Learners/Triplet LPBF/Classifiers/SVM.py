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
from sklearn.svm import SVC
from sklearn import metrics
import os


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from numpy import mean
from numpy import std


def SVM(X_train, X_test, y_train, y_test, classes, total_path):

    print('Model to be trained is SVM')

    random_state = np.random.RandomState(0)
    #svc_model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',decision_function_shape='ovo', verbose=True,random_state=None)
    svc_model = SVC(kernel='rbf', probability=True, random_state=random_state)
    model = svc_model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    predictions = model.predict(X_test)
    print("SVM Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'SVM'+'_without normalization w/o Opt'
    graph_name2 = 'SVM'

    graph_1 = 'SVM'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'SVM'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=200)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Greens, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')

        # disp.ax_.set_title(title)
        plt.title(title, size=12)

        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', dpi=400)
        plt.show()
    savemodel = 'SVM'+'_model'+'.sav'
    joblib.dump(model, savemodel)

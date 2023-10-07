# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:27:06 2021

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
from xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from numpy import sort

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import os
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder

# %%


def XGBoost(X_train, X_test, y_train, y_test, classes, total_path):

    print('Model to be trained is XGBoost')

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    predictions = model.predict(X_test)
    print("XGBoost Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'XGBoost'+'_without normalization w/o Opt'
    graph_name2 = 'XGBoost'

    graph_1 = 'XGBoost'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'XGBoost'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.pink, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')
        plt.title(title, size=12)
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', dpi=400)
        plt.show()

    savemodel = 'XGBoost'+'_model'+'.sav'
    joblib.dump(model, savemodel)

    plt.figure(figsize=(20, 10), dpi=400)
    plot_importance(model)
    plt.savefig('Feature_XGBoost.png', bbox_inches='tight', dpi=400)
    pyplot.show()

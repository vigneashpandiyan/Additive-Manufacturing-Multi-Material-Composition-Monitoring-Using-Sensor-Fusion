# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:27:06 2021

@author: srpv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
from sklearn import metrics
import collections
import os
from tqdm.notebook import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # implementing train-test-split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns


def LR(X_train, X_test, y_train, y_test, file, folder_created):

    model_classifier = LogisticRegression(max_iter=1000, random_state=123)
    model_classifier.fit(X_train, y_train)

    predictions = model_classifier.predict(X_test)

    pred_prob = model_classifier.predict_proba(X_test)
    y_pred_prob = np.vstack((y_test, predictions)).transpose()

    y_pred_prob = np.hstack((y_pred_prob, pred_prob))

    print("LogisticRegression Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'LR'+'_without normalization w/o Opt'
    graph_name2 = 'Logistic Regression'

    graph_1 = folder_created+'/'+str(file)+'_LR'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = folder_created+'/'+str(file)+'_LR'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)

        cm = confusion_matrix(y_test, predictions, labels=model_classifier.classes_)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classifier.classes_)

        disp.plot()
        plt.title(title, size=12)

        plt.savefig(graphname, bbox_inches='tight', dpi=400)

    savemodel = folder_created+'/'+'LR'+'_model'+'.sav'
    joblib.dump(model_classifier, savemodel)

    return y_pred_prob, pred_prob, model_classifier

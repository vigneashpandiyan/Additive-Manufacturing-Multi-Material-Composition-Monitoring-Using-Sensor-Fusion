"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
#%% Libraries required:

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# %%

def NN(X_train, X_test, y_train, y_test, classes, total_path):
    """
    Trains a neural network model using the MLPClassifier algorithm and evaluates its performance.

    Args:
        X_train (array-like): The training input samples.
        X_test (array-like): The test input samples.
        y_train (array-like): The target values for the training samples.
        y_test (array-like): The target values for the test samples.
        classes (array-like): The class labels.
        total_path (str): The path to save the generated graphs and model.

    Returns:
        None
    """

    print('Model to be trained is NN')

    model = MLPClassifier(hidden_layer_sizes=(60, 40, 20), max_iter=50000, validation_fraction=0.1)
    model.fit(X_train, y_train)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)

    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    predictions = model.predict(X_test)
    print("NN Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'NN'+'_without normalization w/o Opt'
    graph_name2 = 'Neural Network'

    graph_1 = 'NN'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2 = 'NN'+'_Confusion_Matrix'+'_'+'Opt'+'.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=classes,
                                                     cmap=plt.cm.Reds, xticks_rotation='vertical',
                                                     normalize=normalize, values_format='0.2f')
        plt.title(title, size=12)
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', dpi=400)
        plt.show()

    savemodel = 'NN'+'_model'+'.sav'
    joblib.dump(model, savemodel)

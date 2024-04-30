# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
from Utils import *
from Network import *
from Dataloader import *
from Loss import *
from Visualization import *
from matplotlib import animation
import os


def generalization(Material_1, Material_2, windowsize, total_path, model, device):
    """
    Perform generalization analysis on two materials.

    Args:
        Material_1 (str): Name of the first material.
        Material_2 (str): Name of the second material.
        total_path (str): Path to the data.
        model: The trained model.
        device: The device to run the model on.

    Returns:
        tuple: A tuple containing the training results, test results, training labels, and test labels.
    """

    folder_created = os.path.join('Figures/', (str(Material_1) + str(Material_2)))
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    rawspace_1, classspace = data_pipeline(Material_1, total_path, windowsize)
    rawspace_2, classspace = data_pipeline(Material_2, total_path, windowsize)
    rawspace = np.stack((rawspace_1, rawspace_2), axis=2)

    X_train, X_test, y_train, y_test = train_test_split(
        rawspace, classspace, test_size=0.30, random_state=123)

    index_train = pd.DataFrame(y_train)
    index_test = pd.DataFrame(y_test)

    train_results, train_labels = compute_embeddings(model, X_train, y_train, index_train, device,
                                                     (str(Material_1) + str(Material_2)) + '_train', folder_created)
    test_results, test_labels = compute_embeddings(model, X_test, y_test, index_test, device,
                                                   (str(Material_1) + str(Material_2)) + '_test', folder_created)

    graph_name_2D = folder_created + '/' + \
        (str(Material_1) + str(Material_2)) + '_Feature_2D' + '.png'
    plot_embeddings(test_results, test_labels, graph_name_2D)

    graph_name_2D = folder_created + '/' + \
        (str(Material_1) + str(Material_2)) + '_Feature_2D_reduced' + '.png'
    plot_embeddings_reduced(test_results, test_labels, graph_name_2D, test_size=0.60)

    graph_name_3D = folder_created + '/' + \
        (str(Material_1) + str(Material_2)) + '_Feature_3D' + '.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created + '/' + (str(Material_1) + str(Material_2)) + '_Tsne_3D' + '.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    graph_name_3D = folder_created + '/' + \
        (str(Material_1) + str(Material_2)) + '_Feature_3D_reduced' + '.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created + '/' + \
        (str(Material_1) + str(Material_2)) + '_Tsne_3D_reduced' + '.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    return train_results, test_results, train_labels, test_labels

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
from Classifier import *
from Visualization import *
from matplotlib import animation


def generalization(Material_1, Material_2, trainset, testset, folder_created, model, device):

    # Compute embeddings for the trainset and testset
    train_results, train_labels = compute_embeddings(
        model, trainset, device, (str(Material_1)+str(Material_2))+'_train', folder_created)
    test_results, test_labels = compute_embeddings(
        model, testset, device, (str(Material_1)+str(Material_2))+'_test', folder_created)

    graph_name_2D = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Feature_2D'+'.png'
    plot_embeddings(test_results, test_labels, graph_name_2D)

    graph_name_2D = folder_created+'/' + \
        (str(Material_1)+str(Material_2))+'_Feature_2D_reduced'+'.png'
    plot_embeddings_reduced(test_results, test_labels, graph_name_2D, test_size=0.60)

    graph_name_3D = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Feature_3D' + '.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Tsne_3D'+'.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    graph_name_3D = folder_created+'/' + \
        (str(Material_1)+str(Material_2))+'_Feature_3D_reduced'+'.png'
    ax, fig = TSNEplot(test_results, test_labels, graph_name_3D,
                       test_size=0.60, ang=115, perplexity=10)

    graph_name = folder_created+'/'+(str(Material_1)+str(Material_2))+'_Tsne_3D_reduced'+'.gif'

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(graph_name, writer=animation.PillowWriter(fps=20))

    return train_results, test_results, train_labels, test_labels

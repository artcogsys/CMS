import os
import numpy as np
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from world.base import *
from brain.monitor import *


def movie(snapshots, data_iter, agent, function, name, dpi=100, fps=1):

    # set monitor for snapshot
    monitor = Monitor()

    # get video writer
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie', artist='CCNLab')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()

    fig.set_size_inches(10, 10)

    # save frames
    with writer.saving(fig, name, dpi):

        for i in range(len(snapshots)):

            # load snapshot
            agent.model.load(snapshots[i])

            agent.add_monitor(monitor)

            world = World(agent)

            # run some test data
            world.test(data_iter)

            function(agent.monitor)

            monitor.reset()

            writer.grab_frame()

            plt.clf()

    plt.close()


def save_plot(fig, out, name):
    """ Helper function to save plot

    :param fig: figure to plot
    :param out: output folder
    :param name: file name
    :return:
    """
    try:
        os.makedirs(out)
    except OSError:
        pass

    fig.savefig(os.path.join(out, name + '.png'))

def accuracy(Y, T):
    """Compute classification accuracy

    :param Y: predictions as one-hot vectors
    :param T: targets as integers
    :return: classification accuracy
    """

    clf = np.argmax(Y, axis=1)

    return np.mean(clf == T)

def confusion_matrix(Y, T):
    """Compute confusion matrix

   :param Y: predictions as one-hot vectors or integers
   :param T: targets as one-hot vectors or integers
   :return: confusion matrix with true class as rows and predicted class as columns
   """

    # convert one-hot vectors to integers
    if Y.ndim > 1:
        Y = np.argmax(Y,axis=1)

    if T.ndim > 1:
        T = np.argmax(T, axis=1)

    classes = np.unique(np.vstack([T, Y]))

    n_vars = len(classes)

    # compute count matrix
    count_mat = np.zeros([n_vars, n_vars])
    for i in range(n_vars):

        idx = np.where(T == classes[i])[0]
        for j in range(n_vars):
            count_mat[i, j] = np.sum(Y[idx]==classes[j])

    conf_mat = np.zeros([n_vars, n_vars])
    for i in range(n_vars):
        conf_mat[i] = count_mat[i] / np.sum(count_mat[i])

    return conf_mat
import os
import numpy as np
import scipy.stats as ss
import matplotlib.animation as manimation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from learners.base import Tester
from models.monitor import Monitor

def movie(snapshots, data, model, function, name, dpi=100, fps=1):

    # get video writer
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie', artist='CCNLab')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()

    fig.set_size_inches(10, 10)

    # save frames
    with writer.saving(fig, name, dpi):

        for i in range(len(snapshots)):

            print 'processing snapshot {0} of {1}'.format(i + 1, len(snapshots))

            # load snapshot
            model.load(snapshots[i])

            # set monitor for snapshot
            monitor = Monitor()

            model.set_monitor(monitor)

            # run some test data
            Tester(model, data).run()

            function(model.monitor)

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

   :param Y: predictions as one-hot vectors
   :param T: targets as integers
   :return: confusion matrix
   """

    [n_samples, n_vars] = Y.shape

    # compute count matrix
    count_mat = np.zeros([n_vars, n_vars])
    for i in range(n_vars):

        # get predictions for trials with real class equal to i
        clf = np.argmax(Y[T == i], axis=1)
        for j in range(n_vars):
            count_mat[i, j] = np.sum(clf == j)

    conf_mat = np.zeros([n_vars, n_vars])
    for i in range(n_vars):
        conf_mat[i] = count_mat[i] / np.sum(count_mat[i])

    return conf_mat
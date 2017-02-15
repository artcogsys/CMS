import os
import numpy as np
import scipy.stats as ss

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
import os
import numpy as np
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from world.base import World
from brain.monitor import Monitor

def plot_loss(gfx, idx, losses, labels=None):
    """ Plot losses

    :param gfx: list of [figure handle, axis handle, hl: data handle]
    :param idx: index at which to plot loss
    :param losses: list of loss values
    :param labels: optional list of loss labels
    :return: fig, ax, hl
    """

    if gfx is None:

        fig, ax = plt.subplots()
        ax.set_xlabel('t')
        ax.set_ylabel('loss')
        hl = np.empty(len(losses), dtype=object)
        for i in range(len(losses)):
            hl[i], = ax.plot(idx, losses[i])
        if not labels is None:
           fig.legend(hl, tuple(labels))
        fig.show()

    else:

        fig = gfx[0]
        ax = gfx[1]
        hl = gfx[2]

        x_data = np.vstack([hl[0].get_xdata(), idx])
        for i in range(len(losses)):
            hl[i].set_xdata(x_data)
            y_data = np.vstack([hl[i].get_ydata(), losses[i]])
            hl[i].set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    return [fig, ax, hl]

def movie(snapshots, data_iter, agent, function, name, dpi=100, fps=1):

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

            # set monitor for snapshot
            monitor = Monitor()

            agent.model.set_monitor(monitor)

            world = World(agent)

            # run some test data
            world.test(data_iter)

            function(agent.model.monitor)

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
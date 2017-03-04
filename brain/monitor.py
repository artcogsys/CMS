from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

#####
## Monitor base class

class Monitor(object):

    def __init__(self, names=None, len=np.inf):
        """

        :param names: if defined then the name or list of names indicates keys to store
        :param len: determines the length of the monitor; stores the last len items at most
        """

        self._dict = defaultdict(list)

        self._names = [names] if names is str else names

        self.len = len

    def set(self, name, value):
        """

        :param name: dictionary key
        :param value: dictionary value
        :return:
        """

        if self._names is None or name in self._names:
            if self.len==1:
                self._dict[name] = [value]
            else:
                if self._dict[name] == self.len:
                    self._dict[name] = self._dict[name][1:].append(value)
                else:
                    self._dict[name].append(value)

    def keys(self):
        return self._dict.keys()

    def __getitem__(self, item):
        """Returns dict[name] as numpy array

        :param item: key name
        :return: numpy array
        """

        assert(item in self.keys())
        data = np.asarray(self._dict[item])
        if data.ndim <=2:
            return data.reshape([np.prod(data.shape)], order='F')
        else:
            return data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]), order='F')

    def run(self):
        """Run a function on the fly; implemented by inheriting from this monitor

        :return:
        """
        pass


class Oscilloscope(Monitor):
    """
    Defines an oscilloscope for one or more signals of interest defined in names
    """

    def __init__(self, names=None, len=np.inf):

        # 't' is used to index the time
        super(Oscilloscope, self).__init__(names, len)

        self.fig = []
        self.ax = []

    def run(self):

        # set a separate figure for each oscilloscope

        keys = self.keys()

        if self.fig == []:

            self.hl = [None] * len(keys)

            for i in range(len(keys)):

                f, a = plt.subplots()
                self.fig.append(f)
                self.ax.append(a)

                key = self.keys()[i]

                self.hl[i], = self.ax[i].plot(np.arange(self[key].size), self[key])
                self.ax[i].set_title(key)

                f.show()

        else:

            for i in range(len(keys)):
                key = self.keys()[i]
                self.hl[i].set_xdata(np.arange(self[key].size))
                self.hl[i].set_ydata(self[key])
                self.ax[i].relim()
                self.ax[i].autoscale_view()
                self.fig[i].canvas.draw()
                self.fig[i].canvas.flush_events()
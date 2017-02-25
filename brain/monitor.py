from collections import defaultdict
import numpy as np

class Monitor(object):

    def __init__(self, names=None, append=True):
        """

        :param names: if defined then the name or list of names indicates keys to store
        :param append: if True, adds to list, otherwise assigns this value to dictionary
        """

        self._dict = defaultdict(list)

        self._names = [names] if names is str else names

        self.append = append


    def set(self, name, value):
        """

        :param name: dictionary key
        :param value: dictionary value
        :return:
        """

        if self._names is None or name in self._names:
            if self.append:
                self._dict[name].append(value)
            else:
                self._dict[name] = [value]

    def keys(self):
        return self._dict.keys()

    def get(self, name):
        """Returns dict[name] as numpy array

        :param name: key name
        :return: numpy array
        """

        assert(name in self.keys())
        data = np.asarray(self._dict[name])
        if data.ndim <=2:
            return data.reshape([np.prod(data.shape)], order='F')
        else:
            return data.reshape([np.prod(data.shape[:2])] + list(data.shape[2:]), order='F')

    def run(self):
        """Run a function on the fly; implemented by inheriting from this monitor

        :return:
        """
        pass
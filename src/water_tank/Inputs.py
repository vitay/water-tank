import numpy as np

class StaticInput(object):

    def __init__(self, size):
        self.size = size
        self._value = np.zeros((self.size,))

    def step(self):
        pass

    def set(self, value):
        if isinstance(value, (float, int)):
            self._value = value * np.ones((self.size,))
            return
        if isinstance(value, (list)):
            value = np.array(value)
        if isinstance(value, (np.ndarray)):
            if value.shape[0] != self.size:
                print("The shape of the array must match the size of the StaticInput.")
                raise IndexError
            self._value = value * np.ones((self.size,))
        else:
            print("The only allowed values are float, int or numpy arrays of the right shape.")
            raise IndexError

    def output(self):
        return self._value



class TimeSeriesInput(object):

    def __init__(self, size, loop=True):
        self.size = size
        self.loop = loop
        self._idx = 0
        self._length = 1
        self._value = np.zeros((self.size,))
        self._data = np.zeros((1, self.size))

    def step(self):
        self._value = self._data[self._idx, :] 
        self._idx += 1
        if self._idx >= self._length: # reached the end of the array
            if self.loop:
                self._idx = 0
            else:
                self._idx -= 1

    def set(self, value):
        self._data = np.array(value)
        self._idx = 0
        self._length, size = self._data.shape
        if size != self.size:
            print("The second dimension must match the size of the population.")
            raise IndexError

    def reset(self):
        self._idx = 0
        self._value = self._data[self._idx, :] 

    def output(self):
        return self._value


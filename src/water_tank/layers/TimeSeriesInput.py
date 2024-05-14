import numpy as np
from .Layer import Layer, transfer_functions

class TimeSeriesInput(Layer):
    """
    Dynamic placeholder for series of input vectors.

    Parameters:
        size: size of the input vector.
        loop: defines whether the buffer loops when arriving at the end.
    """

    def __init__(self, size:int, loop:bool=True) -> None:

        self.size = size
        self.loop = loop
        self._idx = 0
        self._length = 1
        self._value = np.zeros((self.size,))
        self._data = np.zeros((1, self.size))

    def step(self) -> None:
        "Reads the next value."
        self._value = self._data[self._idx, :] 
        self._idx += 1
        if self._idx >= self._length: # reached the end of the array
            if self.loop:
                self._idx = 0
            else:
                self._idx -= 1

    def set(self, value:np.ndarray) -> None:
        "Sets the buffer to `value`."
        self._data = np.array(value)
        self._idx = 0
        self._length, size = self._data.shape
        if size != self.size:
            print("The second dimension must match the size of the population.")
            raise IndexError

    def reset(self) -> None:
        "Resets the buffer."
        self._idx = 0
        self._value = self._data[self._idx, :] 

    def output(self) -> np.ndarray:
        """
        Returns:
            a vector of activities.
        """
        return self._value
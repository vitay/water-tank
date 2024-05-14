import numpy as np
from .Layer import Layer, transfer_functions


class StaticInput(Layer):
    """
    Static placeholder for input vectors.

    Parameters:
        size: size of the vector.
    """

    def __init__(self, size:int) -> None:
        self.size = size
        self._value = np.zeros((self.size,))

    def step(self) -> None:
        "Does nothing."
        pass

    def set(self, value:np.ndarray) -> None:
        """
        Sets the value of the vector. The dimensions must match with `self.size`.

        Parameters:
            value: new vector value.
        """
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

    def output(self) -> np.ndarray:
        """
        Returns:
            a vector of activities.
        """
        return self._value
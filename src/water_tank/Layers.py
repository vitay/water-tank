import numpy as np

class Layer(object):
    pass

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


class LinearReadout(object):
    r"""
    Linear readout layer. Performs a weighted sum of its inputs, without dynamics.
    
    $$\mathbf{z} = W^o \times \mathbf{r}$$

    Parameters:
        size: number of neurons.
    """

    def __init__(self, size:int) -> None:
        self.size = size
        self.z = np.zeros((self.size,))

        self.projections = []

    def output(self) -> np.ndarray:
        """
        Returns:
            a vector of activities.
        """
        return self.z

    def step(self, force:np.ndarray=None) -> None:
        """
        Performs one update of the internal variables.

        Parameters:
            force: if not None, force the output to the provided vector.
        """

        self.z = self._collect_inputs()

        if force is None:
            self.z = self.x
        else:
            self.z = force

    def _collect_inputs(self):

        inp = np.zeros(self.size)
        for proj in self.projections:
            inp += proj.step().reshape((self.size,))

        return inp
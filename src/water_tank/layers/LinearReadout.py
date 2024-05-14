import numpy as np
from .Layer import Layer, transfer_functions

class LinearReadout(Layer):
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

        self.x = self._collect_inputs()

        if force is None:
            self.z = self.x
        else:
            self.z = force
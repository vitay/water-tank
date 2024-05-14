import numpy as np

from .Layer import Layer, transfer_functions

class RecurrentLayer(Layer):
    r"""
    Reservoir of recurrently connected neurons.
    
    $$\tau \, \frac{d \mathbf{x}(t)}{dt} + \mathbf{x}(t) = W^\text{in} \times I(t) + W^\text{rec} \times \mathbf{r}(t) + W^\text{fb} \times \mathbf{z}(t)$$
        
    $$\mathbf{r}(t) = f(\mathbf{x}(t))$$


    Parameters:
        size: number of neurons.
        tau: time constant.
        transfer_function: transfer function.
    """

    def __init__(self, size:int, tau:float=10.0, transfer_function:str='tanh') -> None:
        
        self.size = size
        self.tau = tau
        self.transfer_function = transfer_functions[transfer_function]

        # Vectors
        self.x = np.zeros((self.size,))
        self.r = np.zeros((self.size,))

        # Projections
        self.projections = []

    def reset(self) -> None:
        """
        Resets the vectors x and r to 0.
        """
        self.x = np.zeros((self.size,))
        self.r = np.zeros((self.size,))

    def output(self)  -> None:
        """
        Returns:
            a vector of activities.
        """
        return self.r

    def step(self) -> None:
        """
        Performs one update of the internal variables.
        """

        inputs = self._collect_inputs()
        self.x += (inputs - self.x) / self.tau
        self.r = self.transfer_function(self.x)

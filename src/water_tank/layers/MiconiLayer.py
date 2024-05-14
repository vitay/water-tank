import numpy as np

from .Layer import Layer, transfer_functions
from ..random import RandomDistribution, Const, Uniform

class MiconiLayer(Layer):
    r"""
    Reservoir of recurrently connected neurons, with perturbations and sliding mean.
    
    $$\tau \, \frac{d \mathbf{x}(t)}{dt} + \mathbf{x}(t) = W^\text{in} \times I(t) + W^\text{rec} \times \mathbf{r}(t) + W^\text{fb} \times \mathbf{z}(t) + \xi(t)$$
        
    $$\mathbf{r}(t) = f(\mathbf{x}(t))$$

    $$\mathbf{r}_\text{mean}(t) = \alpha \, \mathbf{r}_\text{mean}(t-1) + (1 - \alpha) \, \mathbf{r}(t) $$

    Parameters:
        size: number of neurons.
        tau: time constant.
        transfer_function: transfer function.
        perturbation_frequency: perturbation frequency.
        perturbation_amplitude: perturbation amplitude.
        alpha_mean: alpha window for computing the mean.
        biases: dictionary of neuron indices and value to have bias neurons with a constant firing rate.
    """

    def __init__(self, 
                 size:int, 
                 tau:float=30.0, 
                 transfer_function:str='tanh',
                 perturbation_frequency:float=3.0,
                 perturbation_amplitude:float=16.0,
                 alpha_mean:float = 0.05,
                 biases = {0: 1.0, 1:1.0, 2:-1.0},
                 ) -> None:
        
        self.size = size
        self.tau = tau
        self.transfer_function = transfer_functions[transfer_function]
        self.perturbation_frequency=perturbation_frequency
        self.perturbation_amplitude=perturbation_amplitude
        self.alpha_mean = alpha_mean
        self.biases = biases

        # Vectors
        self.x = np.zeros((self.size,))
        self.r = np.zeros((self.size,))
        self.r_mean = np.zeros((self.size,))

        # Projections
        self.projections = []

    def output(self)  -> np.ndarray:
        """
        Returns:
            a vector of activities (`r`).
        """
        return self.r
    
    def init(self, 
             x: RandomDistribution=Uniform(-0.1, 0.1),
             r_mean: RandomDistribution=Const(0.0),
             ) -> None:
        """
        Initializes `x`, `r` randomly and `r_mean` to zero.
        """
        self.x = x.sample((self.size,))
        self.r = self.transfer_function(self.x)
        # Sources for the bias
        for idx, val in self.biases.items():
            self.r[idx] = val
        self.r_mean = np.zeros((self.size,))

    def step(self, perturbation:np.ndarray=None) -> None:
        """
        Performs one update of the internal variables.

        Parameters:
            perturbation: array of indices of neurons receiving a perturbation.
        """
        # Afferent connections
        inputs = self._collect_inputs()
        
        # Perturbation
        if perturbation is not None:
            noise = np.random.uniform(
                -self.perturbation_amplitude, 
                self.perturbation_amplitude, 
                (self.size,))
            inputs[perturbation] += noise[perturbation]

        # Dynamics
        self.x += (inputs - self.x) / self.tau
        # Firing rate
        self.r = self.transfer_function(self.x)
        # Sources for the bias
        for idx, val in self.biases.items():
            self.r[idx] = val
        # Sliding mean
        self.r_mean = self.alpha_mean * self.r_mean + (1 - self.alpha_mean) * self. r

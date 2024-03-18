import numpy as np

# Allowed transfer functions
transfer_functions = {
    'tanh': np.tanh
}


class Layer(object):
    def output(self)  -> None:
        """
        Returns:
            a vector of activities.
        """
        raise NotImplementedError

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

    def _collect_inputs(self):

        inp = np.zeros(self.size)
        for proj in self.projections:
            inp += proj.step().reshape((self.size,))
        return inp


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
    """

    def __init__(self, 
                 size:int, 
                 tau:float=30.0, 
                 transfer_function:str='tanh',
                 perturbation_frequency=3.0,
                 perturbation_amplitude=16.0,
                 alpha_mean = 0.05,
                 ) -> None:
        
        self.size = size
        self.tau = tau
        self.transfer_function = transfer_functions[transfer_function]
        self.perturbation_frequency=perturbation_frequency
        self.perturbation_amplitude=perturbation_amplitude
        self.alpha_mean = alpha_mean

        # Vectors
        self.x = np.zeros((self.size,))
        self.r = np.zeros((self.size,))
        self.r_mean = np.zeros((self.size,))

        # Projections
        self.projections = []

    def output(self)  -> None:
        """
        Returns:
            a vector of activities.
        """
        return self.r
    
    def init(self) -> None:
        self.x = np.random.uniform(-0.1, 0.1, (self.size,))
        self.r = self.transfer_function(self.x)
        self.r_mean = np.zeros((self.size,))

    def step(self, perturbation=None) -> None:
        """
        Performs one update of the internal variables.
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

        self.x += (inputs - self.x) / self.tau
        self.r = self.transfer_function(self.x)
        self.r_mean = self.alpha_mean * self.r_mean + (1 - self.alpha_mean) * self. r

    def _collect_inputs(self):

        inp = np.zeros(self.size)
        for proj in self.projections:
            inp += proj.step().reshape((self.size,))
        return inp



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

        self.x = self._collect_inputs()

        if force is None:
            self.z = self.x
        else:
            self.z = force

    def _collect_inputs(self):

        inp = np.zeros(self.size)
        for proj in self.projections:
            inp += proj.step().reshape((self.size,))

        return inp
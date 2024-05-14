import numpy as np

class RandomDistribution(object):
    pass

class Const(RandomDistribution):
    """
    Constant "random" distribution, returning the same value.

    Parameters:
        value: constant value.
    """
    def __init__(self, value:float):
        self.value = value

    def sample(self, shape:tuple) -> np.ndarray:
        "Samples from the distribution and returns an array of the desired shape."
        return self.value * np.ones(shape)

class Uniform(RandomDistribution):
    """
    Uniform distribution, returning values between `min` and `max`.

    Parameters:
        min: lower bound.
        max: upper bound.
    """
    def __init__(self, min:float, max:float):
        self.min = min
        self.max = max
        self.rng = np.random.default_rng()

    def sample(self, shape:tuple) -> np.ndarray:
        "Samples from the distribution and returns an array of the desired shape."
        return self.rng.uniform(self.min, self.max, shape)

class Normal(RandomDistribution):
    """
    Normal distribution, returning values with a mean of `mean` and a standard deviation of `std`.

    Parameters:
        mean: mean.
        std: standard deviation.
    """
    def __init__(self, mean:float, std:float):
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng()

    def sample(self, shape:tuple) -> np.ndarray:
        "Samples from the distribution and returns an array of the desired shape."
        return self.rng.normal(self.mean, self.std, shape)

class Bernouilli(RandomDistribution):
    """
    Bernouilli (binomial) distribution, returning the first of the two values with probability $p$.

    Parameters:
        values: list of values.
        p: probability of returning the first value.
    """
    def __init__(self, values:list, p:float=0.5):
        self.values = values
        self.p = p
        self.rng = np.random.default_rng()

    def sample(self, shape:tuple) -> np.ndarray:
        "Samples from the distribution and returns an array of the desired shape."
        return self.rng.choice(self.values, size=shape, p=[self.p, 1-self.p])

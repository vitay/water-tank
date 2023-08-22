import numpy as np

class RandomDistribution(object):
    pass

class Const(RandomDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self, shape):
        return self.value * np.ones(shape)

class Uniform(RandomDistribution):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.rng = np.random.default_rng()

    def sample(self, shape):
        return self.rng.uniform(self.min, self.max, shape)

class Normal(RandomDistribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng()

    def sample(self, shape):
        return self.rng.normal(self.mean, self.std, shape)

class Bernouilli(RandomDistribution):
    def __init__(self, values, p=0.5):
        self.values = values
        self.p = p
        self.rng = np.random.default_rng()

    def sample(self, shape):
        return self.rng.choice(self.values, size=shape, p=[self.p, 1-self.p])

import numpy as np

from .Layers import StaticInput, RecurrentLayer, LinearReadout
from .Projections import connect
from .RandomDistributions import *
from .LearningRules import RLS
from .Recorders import Recorder
    

class ESN(object):
    """
    Standard Echo-State Network with plasticity in the readout weights. 
    """

    def __init__(self, 
            N:int, 
            N_in:int, 
            N_out:int, 
            g:float, 
            tau:float, 
            sparseness:float
        ):
        self.N = N
        self.N_in = N_in
        self.N_out = N_out

        # Input population
        self.inp = StaticInput(size=N_in)

        # Reservoir 
        self.reservoir = RecurrentLayer(size=N, tau=tau)

        # Readout
        self.readout = LinearReadout(size=N_out)

        # Input projection
        self.inp_reservoir = connect(
            pre = self.inp, 
            post = self.reservoir, 
            weights = Bernouilli([-1.0, 1.0], p=0.5), 
            bias = None,
            sparseness = 0.1
        )

        # Recurrent projection
        self.reservoir_reservoir = connect(
            pre = self.reservoir, 
            post = self.reservoir, 
            weights = Normal(0.0, g/np.sqrt(sparseness*N)), 
            bias = Bernouilli([-1.0, 1.0], p=0.5), # very important
            sparseness = sparseness
        )

        # Readout projection
        self.reservoir_readout = connect(
            pre = self.reservoir, 
            post = self.readout,
            weights = Const(0.0),
            bias = Const(0.0), # learnable bias
            sparseness = 1.0 # readout should be dense
        )

        # Learning rules
        self.learning_rule = RLS(projection=self.reservoir_readout, delta=1e-6)

        # Recorder
        self.recorder = Recorder()

    def train(self, X, Y, warmup:int=0, record:bool=True):

        for t, (x, y) in enumerate(zip(X, Y)): 

            # Inputs/targets
            self.inp.set(x)

            # Steps 
            self.reservoir.step() 
            self.readout.step()

            # Learning
            if t >= warmup: 
                self.learning_rule.step(error=y - self.readout.output())

            # Recording
            if record:
                self.recorder.record({
                    'reservoir': self.reservoir.output(), 
                    'readout': self.readout.output(),
                })

    def force(self, X, Y, warmup:int=0, record:bool=True):

        for t, (x, y) in enumerate(zip(X, Y)): 

            # Inputs/targets
            self.inp.set(x)

            # Steps 
            self.reservoir.step() 
            self.readout.step()

            # Learning
            if t >= warmup: 
                self.learning_rule.step(error= y - self.readout.output())

            # Recording
            if record:
                self.recorder.record({
                    'reservoir': self.reservoir.output(), 
                    'readout': self.readout.output(),
                })
        
    def autoregressive(self, duration:int, record:bool=True):

        for _ in range(duration): 
            # Autoregressive input
            self.inp.set(self.readout.output())  

            # Steps 
            self.reservoir.step() 
            self.readout.step()

            # Recording
            if record:
                self.recorder.record({
                    'reservoir': self.reservoir.output(), 
                    'readout': self.readout.output()
                })
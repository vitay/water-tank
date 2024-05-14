import numpy as np

from ..projections.Projection import Projection
from ..projections.SparseProjection import SparseProjection

class MiconiLearningRule(object):
    r"""
    Miconi learning rule (reward-guided perturbation learning).

    Equation:

    $$TODO$$

    Parameters:
        projection: projection on which to apply the learning rule.
        learning_rate: learning rate.
    """

    def __init__(self, 
                 projection:Projection, 
                 learning_rate:float,
                 clip_dW:float,
                 ):

        self.projection = projection
        self.learning_rate = learning_rate
        self.clip_dW = clip_dW

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection))
        self._has_bias = self.projection._has_bias

        self.trace = self.projection._empty_copy()

    def step(self):
        """
        Update the trace. To be called at each time step.
        """

        if not self.sparse:
            self.trace += np.outer(
                np.power(self.projection.post.r - self.projection.post.r_mean, 3), 
                self.projection.pre.output()
            )
        else:
            self.trace += self.projection.W.outer(
                np.power(self.projection.post.r - self.projection.post.r_mean, 3), 
                self.projection.pre.output()
            )

    def train(self, reward:float, critic:float=0.0):
        """
        Modifies the weights based on the reward, the critic (average reward) and the trace.
        """

        if not self.sparse:

            dW = self.learning_rate * self.trace * (reward - critic) * critic

            self.projection.W += dW.clip(min=-self.clip_dW, max=self.clip_dW),

            self.trace.fill(0.0)

        else:
            dW = []
            db = []

            #for i in range(self.nb_post):
            #    r = self.projection.input(i).reshape((-1, ))
            #    dW.append(self.learning_rate * advantage[i] * r)
            #    db.append(self.learning_rate * advantage[i])
        
            #self.projection._update_parameters(dW, db if self._has_bias else None)

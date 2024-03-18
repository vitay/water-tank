import numpy as np

from .Projections import Projection, SparseProjection

class DeltaLearningRule(object):
    r"""
    Delta learning rule (online linear regression).

    Equation:

    $$\Delta W = \eta \, (\mathbf{t} - \mathbf{y}) \times \mathbf{x}^T$$

    Parameters:
        projection: projection on which to apply the learning rule.
        learning_rate: learning rate.
    """

    def __init__(self, projection:Projection, learning_rate:float):

        self.projection = projection
        self.learning_rate = learning_rate

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection))
        self._has_bias = self.projection._has_bias

    def train(self, target):

        error = target - self.projection.post.output()

        if not self.sparse:

            self.projection._update_parameters(
                weights = self.learning_rate * np.outer(error, self.projection.pre.output()),
                bias=self.learning_rate * error if self._has_bias else None,
            )

        else:
            dW = []
            db = []

            for i in range(self.nb_post):
                r = self.projection.input(i).reshape((-1, ))
                dW.append(self.learning_rate * error[i] * r)
                db.append(self.learning_rate * error[i])
        
            self.projection._update_parameters(dW, db if self._has_bias else None)

class RLS(object):
    """
    Recursive least-squares (RLS) learning rule for FORCE learning.

    Parameters:
        projection: projection on which to apply the learning rule.
        delta: initial diagonal value of the correlation matrix.
    """

    def __init__(self, projection:Projection, delta:float=1e-6) -> None:

        self.projection = projection
        self.delta = delta

        self.nb_post = self.projection.post.size

        self.sparse = isinstance(self.projection, (SparseProjection))

        self._has_bias = self.projection._has_bias

        self.P = [
            np.identity(
                self.projection.nb_connections(i) + (1 if self._has_bias else 0)
            ) / self.delta 
            for i in range(self.nb_post)
        ]
        

    def train(self, error):

        # Apply the FORCE learning rule to the readout weights
        dW = []
        db = []

        for i in range(self.nb_post):

            r = self.projection.input(i).reshape((-1, ))
            
            if self._has_bias:
                r = np.concatenate((r, np.ones((1, ))))

            # Multiply the rates with the inverse correlation matrix P*R
            PxR =  self.P[i] @ r

            # Normalization term 1 + R'*P*R
            RxPxR =  (r.T @  PxR + 1.)
            
            # Update the inverse correlation matrix P <- P - ((P*R)*(P*R)')/(1+R'*P*R)
            self.P[i] -= np.outer(PxR, PxR)/RxPxR
            
            # Learning rule W <- W + e * (P*R)/(1+R'*P*R)
            diff = error[i] * (PxR/RxPxR).reshape((-1,))

            if self._has_bias:
                dW.append(diff[:-1])
                db.append(diff[-1])
            else:
                dW.append(diff)

        self.projection._update_parameters(dW, db)



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
                self.projection.post.r - self.projection.post.r_mean, 
                self.projection.pre.output()
            )
        else:
            self.trace = 1.0

    def train(self, reward:float, critic:float=0.0):
        """
        Modifies the weights based on the reward, the critic (average reward)
        """

        advantage = reward - critic

        if not self.sparse:

            dW = self.learning_rate * self.trace * advantage * critic

            self.projection._update_parameters(
                weights = dW.clip(min=-self.clip_dW, max=self.clip_dW),
                bias=None, # TODO
            )

            self.trace.fill(0.0)

        else:
            dW = []
            db = []

            #for i in range(self.nb_post):
            #    r = self.projection.input(i).reshape((-1, ))
            #    dW.append(self.learning_rate * advantage[i] * r)
            #    db.append(self.learning_rate * advantage[i])
        
            #self.projection._update_parameters(dW, db if self._has_bias else None)

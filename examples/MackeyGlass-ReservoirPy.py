import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, RLS
from reservoirpy.datasets import mackey_glass

# Mackey-Glass chaotic time series
X = mackey_glass(n_timesteps=2000)
X =  2.0 * (X - X.min()) / (X.max() - X.min()) - 1.0

# Online Reservoir
reservoir = Reservoir(units=200, lr=0.3, sr=1.25)
readout = RLS(output_dim=1)
esn = reservoir >> readout

# Train / test
d_train = 500
d_test = 1000

# Supervised training
training = esn.train(X[:d_train], X[1:1+d_train])

# Autoregression
predictions = []
errors = list(X[1:1+d_train] - training)
pred = [X[d_train-1]]
for t in range(d_test):
    pred = esn.run(pred)
    predictions.append(pred[0])
    errors.append(X[d_train + t + 1, 0] - pred[0])


plt.figure()
plt.subplot(211)
plt.plot(training, label='prediction (train)')
plt.plot(X[1:1+d_train], label='ground truth (train)')
plt.plot(np.arange(1+d_train, 1+d_train+d_test), predictions, label='prediction (test)')
plt.plot(np.arange(1+d_train, 1+d_train+d_test), X[1+d_train: 1+d_train+d_test, 0], label='ground truth (test)')
plt.legend()
plt.subplot(212)
plt.plot(errors, label='error')
plt.legend()

plt.show()
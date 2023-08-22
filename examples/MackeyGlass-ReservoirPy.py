import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge, RLS
from reservoirpy.datasets import mackey_glass
from reservoirpy.observables import rmse, rsquare

X = mackey_glass(n_timesteps=2000)
print(X.shape)
reservoir = Reservoir(units=200, lr=0.3, sr=1.25)
readout = RLS(output_dim=1)

esn = reservoir >> readout

d_train = 500
d_test = 1000
training = esn.train(X[:d_train], X[1:1+d_train])

predictions = []

inp = [X[d_train-1]]
for t in range(d_test):
    pred = esn.run(inp)
    inp = pred.copy()
    predictions.append(pred[0])

plt.figure()
plt.plot(training, label='prediction')
plt.plot(X[1:1+d_train], label='ground truth')
plt.plot(np.arange(1+d_train, 1+d_train+d_test), predictions, label='prediction')
plt.plot(np.arange(1+d_train, 1+d_train+d_test), X[1+d_train: 1+d_train+d_test, 0], label='ground truth')
plt.legend()

plt.show()
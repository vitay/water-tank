import numpy as np
import matplotlib.pyplot as plt
import water_tank as wt
from reservoirpy.datasets import mackey_glass, lorenz

N_in = 1 # number of inputs
N_out = 1 # number of outputs
N = 200 # number of neurons
g = 1.25 # scaling factor
tau = 3.3 # time constant
sparseness = 0.1 # sparseness of the recurrent weights


class RC(object):

    def __init__(self, N, N_in, N_out, g, tau, sparseness):

        # Input population
        self.inp = wt.StaticInput(size=N_in)

        # Reservoir 
        self.rc = wt.TanhReservoir(N, g, tau, sparseness=sparseness)

        # Readout
        self.readout = wt.LinearReadout(size=N_out)

        # Target population
        self.target = wt.StaticInput(size=N_out)

        # Connections
        #self.inp_proj = self.rc.connect(self.inp, wt.Uniform(-1.0, 1.0))
        self.inp_proj = self.rc.connect(self.inp, wt.Bernouilli([-1.0, 1.0], p=0.5))

        self.readout_proj = self.readout.connect(self.rc, wt.Const(0.0), bias=0.0, sparseness=1.0)

        #self.feedback_proj = self.rc.connect(self.readout,  wt.Uniform(-1.0, 1.0))

        # Learning rules
        #self.learningrule = wt.DeltaLearningRule(projection=self.readout_proj, target=self.target, learning_rate=0.001)
        self.learningrule = wt.RLS(projection=self.readout_proj, target=self.target, delta=1e-6)

    @wt.measure
    def train(self, X, Y, warmup=0):
        recordings = {'rc': [], 'readout': [], 'target': []}

        for t, (x, y) in enumerate(zip(X, Y)): 
            # Inputs/targets
            self.inp.set(x)  
            self.target.set(y)

            # Steps 
            self.rc.step() 
            self.readout.step()

            # Learning
            if t >= warmup: self.learningrule.step()

            # Recording
            recordings['rc'].append(self.rc.output())
            recordings['target'].append(self.target.output())
            recordings['readout'].append(self.readout.output())

        recordings['rc'] = np.array(recordings['rc'])
        recordings['target'] = np.array(recordings['target'])
        recordings['readout'] = np.array(recordings['readout'])
        
        return recordings
    
    @wt.measure
    def autoregressive(self, X, Y, warmup=0):
        recordings = {'rc': [], 'readout': [], 'target': []}

        for t, (x, y) in enumerate(zip(X, Y)): 
            # Inputs/targets
            if t < warmup:
                self.inp.set(x)  
            else:
                self.inp.set(self.readout.output())  

            # Steps 
            self.rc.step() 
            self.readout.step()

            # Recording
            recordings['rc'].append(self.rc.output())
            recordings['target'].append(y)
            recordings['readout'].append(self.readout.output())

        recordings['rc'] = np.array(recordings['rc'])
        recordings['target'] = np.array(recordings['target'])
        recordings['readout'] = np.array(recordings['readout'])
        
        return recordings
    
# Create network
net = RC(N, N_in, N_out, g, tau, sparseness)

# Define input / target
T = 2000
mg = mackey_glass(T)
#mg = 2 * (mg - mg.min()) / (mg.max() - mg.min()) - 1.
forecast = 1
X = mg[:-forecast, :]
Y = mg[forecast:, :]


"""
T = 10000
mg = lorenz(T)
mg[:, 0] = 2 * (mg[:, 0] - mg[:, 0].min()) / (mg[:, 0].max() - mg[:, 0].min()) - 1.
mg[:, 1] = 2 * (mg[:, 1] - mg[:, 1].min()) / (mg[:, 1].max() - mg[:, 1].min()) - 1.
mg[:, 2] = 2 * (mg[:, 2] - mg[:, 2].min()) / (mg[:, 2].max() - mg[:, 2].min()) - 1.
forecast = 1
X = mg[:-forecast, :]
Y = mg[forecast:, :]

"""

# Simulate
d_train = 500
d_test = 1000
data_train = net.train(X[:d_train, :], Y[:d_train, :], warmup=0)
data_test = net.autoregressive(X[d_train:d_train+d_test, :], Y[d_train:d_train+d_test, :], warmup=0)

print(net.learningrule.projection.bias)


# Visualize
plt.figure()
plt.plot(data_train['target'][:, 0], label='ground truth (training)')
plt.plot(data_train['readout'][:, 0], label='prediction (training)')
plt.plot(np.linspace(d_train, d_train+d_test, d_test), data_test['target'][:, 0], label='ground truth (test)')
plt.plot(np.linspace(d_train, d_train+d_test, d_test), data_test['readout'][:, 0], label='prediction (test)')
plt.plot(np.linspace(d_train, d_train+d_test, d_test), data_test['target'][:, 0] - data_test['readout'][:, 0], label='error')
plt.title("Autoregression")
plt.legend()

plt.show()




import numpy as np

class Recorder(object):

    def __init__(self):

        self.data = {}

    def record(self, data):

        if not isinstance(data, (dict,)):
            print("Recorder.record(): only dictionaries can be recorded.")

        for key, val in data.items():
            if not key in self.data.keys():
                self.data[key] = []
            self.data[key].append(val)

    def get(self):

        res = {}
        for key, val in self.data.items():
            try:
                res[key] = np.array(val)
            except:
                res[key] = val
        self.clear()
        return res
    
    def clear(self):
        self.data = {}
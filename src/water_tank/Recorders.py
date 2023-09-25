import numpy as np

class Recorder(object):
    r"""
    Structure to record activities during a simulation.

    Example:

    ```python
    recorder = Recorder()
    for t in range(1000):
        # ...
        recorder.record({
            'rc': rc.output(), 
            'readout': readout.output(),
        })
    data = recorder.get()
    ```
    """

    def __init__(self) -> None:

        self.data = {}

    def record(self, data:dict) -> None:
        """
        Accumulates the content of the dictionary. Keys are created if needed.

        Parameters:
            data: dictionary of numpy arrays to record.
        """

        if not isinstance(data, (dict,)):
            print("Recorder.record(): only dictionaries can be recorded.")

        for key, val in data.items():
            if not key in self.data.keys():
                self.data[key] = []
            self.data[key].append(val)

    def get(self) -> dict:
        """
        Returns:
            a dictionary of numpy arrays.
        """

        res = {}
        for key, val in self.data.items():
            try:
                res[key] = np.array(val)
            except:
                res[key] = val
        self.clear()
        return res
    
    def clear(self):
        "Clears the data."
        self.data = {}
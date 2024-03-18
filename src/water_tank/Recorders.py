import numpy as np

class Recorder(object):
    r"""
    Data structure to record activities during a simulation.

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
        self._to_clear = False

    def record(self, data:dict) -> None:
        """
        Accumulates the content of the dictionary. Keys are created if needed.

        Parameters:
            data: dictionary of numpy arrays to record.
        """
        if self._to_clear : self.clear()

        if not isinstance(data, (dict,)):
            print("Recorder.record(): only dictionaries can be recorded.")

        for key, val in data.items():
            # Create a key if it is the first time
            if not key in self.data.keys():
                self.data[key] = []
            # Check the value
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val[0]
            # Append it to the list
            self.data[key].append(val)

    def get(self) -> dict:
        """
        Returns a dictionary with the recorded data.

        The next call to record() with clear the data.

        Returns:
            a dictionary of numpy arrays.
        """

        # Return numpy arrays when possible
        res = {}
        for key, val in self.data.items():
            try:
                res[key] = np.array(val)
            except:
                res[key] = val
        # Mark the data as to-be-cleared
        self._to_clear = True

        return res
    
    def clear(self):
        "Clears the data."
        self._to_clear = False
        self.data = {}
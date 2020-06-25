import numpy as np
import pandas as pd

from pysptools.eea import FIPPI, PPI, NFINDR, ATGP
from libpyhat.data.spectra import Spectra

class EMI(object):
    def __init__(self, data, method):
        self.supported_formats = ("FIPPI", "PPI", "NFINDR", "ATGP")
        self.method = method
        self.data = data
        self.endmembers = None

    def method():
        doc = "The method used for end-member identification."

        def fget(self):
            return self._method

        def fset(self, value):
            try:
                self._method = globals()[value.upper()]()
            except KeyError:
                raise NotImplementedError(f"Method '{value}' is not currently supported.  "\
                                          f"Supported formats are {self.supported_formats}")

        def fdel(self):
            del self._method

        return locals()
    method = property(**method())

    def data():
        doc = "Hyperspectral cube formatted as a numpy.ndarray."
        def fget(self):
            return self._data

        def fset(self, value):
            if issubclass(type(value), pd.DataFrame):
                self._data = value.to_numpy()
            elif issubclass(type(value), np.ndarray):
                self._data = value
            else:
                raise ValueError(f"Data must inherit from pd.DataFrame or np.ndarray")
            if len(self._data.shape) == 2:
                self._data = np.expand_dims(self._data, 0)

        def fdel(self):
            del self._data

        return locals()
    data = property(**data())

    def endmembers():
        doc = "Endmembers identified by the algorthm"

        def fget(self):
            if self._endmembers is None:
                print("No endmembers found. Try running .extract() before accessing endmembers.")
            return self._endmembers

        def fset(self, value):
            if issubclass(type(value), pd.DataFrame):
                self._endmembers = value.to_numpy()
            elif issubclass(type(value), np.ndarray):
                self._endmembers = value
            elif isinstance(value, type(None)):
                self._endmembers = None
            else:
                raise ValueError(f"Endmembers must inherit from pd.DataFrame or np.ndarray")

        def fdel(self):
            del self._endmembers

        return locals()
    endmembers = property(**endmembers())

    def extract(self, **kwargs):
        self.endmembers = self.method.extract(self.data, **kwargs)
        return self.endmembers

    def display(self, **kwargs):
        self.method.display(**kwargs)

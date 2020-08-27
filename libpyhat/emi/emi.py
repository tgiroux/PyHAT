import pandas as pd
import numpy as np
from pysptools.eea import FIPPI, PPI, NFINDR, ATGP
from libpyhat.data.utils import tabular_to_cube

def emi(data, emi_method, **kwargs):
    supported_methods = ("FIPPI", "PPI", "NFINDR", "ATGP")
    try:
        if emi_method.upper() in supported_methods:
            method = globals()[emi_method.upper()]()
        else:
            print(f"{emi_method} is not a supported method.  Supported methods are {supported_methods}")
            return 1
    except KeyError:
        print(f"Unable to instantiate class from {emi_method}.")
        return 1

    spectra = tabular_to_cube(data)

    endmembers = method.extract(spectra, **kwargs)
    endmember_indices = [i[0] for i in method.get_idx()]
    indices = np.zeros(data.shape[0], dtype=int)
    indices[endmember_indices] = 1
    return endmember_indices

import numpy as np
import pandas as pd
from pysptools.noise import MNF
from libpyhat import Spectra
from libpyhat.data.utils import tabular_to_cube

def mnf(data):
    '''
    Description: Minimum Noise Fraction (MNF) wrapper for pysptools implementation
    Rationale: Removes noise while preserving information
    '''
    return MNF().apply(tabular_to_cube(data))

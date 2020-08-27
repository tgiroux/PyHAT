import numpy as np
def tabular_to_cube(tab_data):
    '''
    Extracts data from tabular form into an (MxNxP) HSI cube ndarray
    ---
    tab_data : pandas DataFrame internally containing data in the shape ((MxN)xP)
    '''
    # get col index where actual data starts
    wvl_index = tab_data.columns.to_numpy().tolist().index('wvl') + 1
    row_index = tab_data.values[0].tolist().index('row')
    col_index = tab_data.values[0].tolist().index('col')

    # get m n p of cube
    m = tab_data.values[-1, row_index]
    n = tab_data.values[-1, col_index]
    p = tab_data.values.shape[1] - wvl_index

    vals = tab_data.values[1:,wvl_index:].astype(float)
    return np.reshape(vals,(m,n,p))

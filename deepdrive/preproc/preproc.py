import numpy as np
from molecules.utils import triu_to_full


def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae.

    Parameters
    ----------
    cm_data_lists : list
        list containing h5py dataset objects of contact matrices

    """
    cm_all = np.hstack(cm_data_lists)

    # Transfer upper triangle to full matrix
    cm_data_full = np.array(list(map(triu_to_full, cm_all.T)))

    # Padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x % 2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # Reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input

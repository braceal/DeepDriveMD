import h5py
import numpy as np
# TODO: update import after molecules bug is fixed
#from molecules.utils import triu_to_full

def triu_to_full(cm0): 
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0 
    cm_full.T[iu1] = cm0 
    np.fill_diagonal(cm_full, 1)
    
    return cm_full

def open_h5(h5_file):
    """
    Opens file in single write multiple reader mode
    libver specifies newest available version,
    may not be backwards compatable

    Parameters
    ----------
    h5_file : str
        name of h5 file to open

    Returns
    -------
    open h5py file

    """
    return h5py.File(h5_file, 'r', libver='latest', swmr=True)


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
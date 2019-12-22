import os
import h5py
import click
import warnings
import numpy as np 
from glob import glob
from contextlib import ExitStack
from utils import cm_to_cvae, open_h5


def validate_sim_path(ctx, param, value):
    """
    Adds abspath to non-None file
    """
    if value:
        return os.path.abspath(value)
    else:
        warnings.warn('No input dirname given, using current directory...')
        return os.path.abspath('.')


@click.command()
@click.option('-f', '--sim_path', default=None,
              callback=validate_sim_path,
              help='Input: OpenMM simulation path')
@click.option('-o', '--out_path', default=None,
              help='Output: CVAE 2D contact map h5 input file')
def main(sim_path, out_path):
    # TODO: generalize omm*/*_cm.h5
    cm_filepath = os.path.join(sim_path, 'omm*/*_cm.h5')

    cm_files = sorted(glob(cm_filepath)) 
    if not cm_files: 
        raise IOError(f'No h5 file found, recheck your input filepath {sim_path}')

    with ExitStack() as stack:
        # Open all h5 files and add them to exit stack
        open_cm_files = map(lambda file: stack.enter_context(open_h5(file)), 
                            cm_files)

        # Iterate through open h5 files and get contact_map datasets 
        cm_data = list(map(lambda file: file['contact_maps'], open_cm_files))
        
        # Compress all .h5 files into one in cvae format 
        cvae_input = cm_to_cvae(cm_data_lists)

        # Create .h5 as cvae input
        cvae_input_file = 'cvae_input.h5'

        # Create and open contact map aggregation output file
        cvae_input_file = stack.enter_context(h5py.File(cvae_input_file, 'w'))

        # Write aggregated contact map dataset to file
        cvae_input_file.create_dataset('contact_maps', data=cvae_input)

if __name__ == '__main__':
    main()
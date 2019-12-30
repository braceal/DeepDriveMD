import os
import h5py
import click
import warnings
from glob import glob
from contextlib import ExitStack
from molecules.utils import open_h5
from deepdrive.preproc import cm_to_cvae
from deepdrive.utils.validators import validate_path


@click.command()
@click.option('-i', '--sim_path', required=True,
              callback=validate_path,
              help='OpenMM simulation path containing output-cm-*.h5 files')
@click.option('-o', '--out', required=True,
              callback=validate_path,
              help='2D contact map h5 file')
def main(sim_path, out):

    # Define wildcard path to contact matrix data
    cm_filepath = os.path.join(sim_path, 'output-cm-*.h5')

    # Collect contact matrix file names sorted by sim_id
    cm_files = sorted(glob(cm_filepath), 
                      key=lambda path: int(path.split('.h5')[-2].split('-')[-1]))
    if not cm_files: 
        raise FileNotFoundError(f'No h5 files found, recheck your input path {sim_path}')

    with ExitStack() as stack:
        # Open all h5 files and add them to exit stack
        open_cm_files = map(lambda file: stack.enter_context(open_h5(file)), 
                            cm_files)

        # Iterate through open h5 files and get contact_map datasets 
        cm_data = list(map(lambda file: file['contact_maps'], open_cm_files))
        
        # Compress all .h5 files into one in cvae format 
        cvae_input = cm_to_cvae(cm_data)

        # Create .h5 as cvae input
        cvae_input_file = os.path.join(out, 'cvae-input.h5')

        # Create and open contact map aggregation output file
        cvae_input_file = stack.enter_context(h5py.File(cvae_input_file, 'w'))

        # Write aggregated contact map dataset to file
        cvae_input_file.create_dataset('contact_maps', data=cvae_input)

if __name__ == '__main__':
    main()
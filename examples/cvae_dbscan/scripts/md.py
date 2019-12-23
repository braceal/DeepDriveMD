import os
import click
import simtk.unit as u
from deepdrive.md import openmm_simulate_amber_fs_pep

# TODO: determine  default type for report and length


def validate_file(ctx, param, value):
    """
    Adds abspath to non-None file
    """
    if value:
        return os.path.abspath(value)

def validate_len(ctx, param, value):
    """
    Checks that length is greater than 0
    """
    if value < 0:
        raise click.BadParameter(f'must be greater than 0, currently {value}')


@click.command()
@click.option('-p', '--pdb', required=True,
              callback=validate_file, help='PDB file')
@click.option('-t', '--topol', default=None, 
              callback=validate_file, help='Topology file')
@click.option('-c', '--chk', default=None,
              callback=validate_file,
              help='Checkpoint file to restart simulation')
@click.option('-l', '--len', 'length', default=10, type=float,
              callback=validate_len,
              help='How long (ns) the system will be simulated')
@click.option('-r', '--report', default=50, type=int,
              callback=validate_len,
              help= 'Time interval (ps) between reports')
@click.option('-g', '--gpu', default=0, type=int, 
              help='ID of gpu to use for the simulation')
def main(pdb, topol, chk, length, report, gpu):
    
    openmm_simulate_amber_fs_pep(pdb,
                                 check_point = chk,
                                 GPU_index=gpu,
                                 output_traj='output.dcd',
                                 output_log='output.log',
                                 output_cm='output_cm.h5',
                                 report_time=report * u.picoseconds,
                                 sim_time=length * u.nanoseconds)

if __name__ == '__main__':
    main()

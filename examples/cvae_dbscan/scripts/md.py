import os
import click
import simtk.unit as u
from deepdrive.md import openmm_simulate_amber_fs_pep
from deepdrive.utils.validators import validate_positive


@click.command()
@click.option('-p', '--pdb', 'pdb_path' required=True,
              type=click.Path(exists=True),
              help='PDB file')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for MD simulation data')

@click.option('-i', '--sim_id', required=True,
              help='Simulation ID for file naming')

@click.option('-t', '--topol', default=None, 
              type=click.Path(exists=True),
              help='Topology file')

@click.option('-c', '--chk', default=None,
              type=click.Path(exists=True),
              help='Checkpoint file to restart simulation')

@click.option('-l', '--len', 'length', default=10, type=float,
              callback=validate_positive,
              help='How long (ns) the system will be simulated')

@click.option('-r', '--report', default=50, type=float,
              callback=validate_positive,
              help= 'Time interval (ps) between reports')

@click.option('-g', '--gpu', default=0, type=int,
              callback=validate_positive,
              help='ID of gpu to use for the simulation')

def main(pdb_path, out_path, sim_id, topol, chk, length, report, gpu):
    openmm_simulate_amber_fs_pep(pdb_path,
                                 checkpnt=chk,
                                 GPU_index=gpu,
                                 checkpnt_fname=os.path.join(out_path, f'checkpnt-{sim_id}.chk'),
                                 output_traj=os.path.join(out_path, f'output-{sim_id}.dcd'),
                                 output_log=os.path.join(out_path, f'output-{sim_id}.log'),
                                 output_cm=os.path.join(out_path, f'output-cm-{sim_id}.h5'),
                                 report_time=report * u.picoseconds,
                                 sim_time=length * u.nanoseconds)

if __name__ == '__main__':
    main()

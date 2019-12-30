import os
import click
import simtk.unit as u
from deepdrive.md import openmm_simulate_amber_fs_pep
from deepdrive.utils.validators import validate_path, validate_positive


@click.command()
@click.option('-p', '--pdb', required=True,
              callback=validate_path, help='PDB file')
@click.option('-o', '--out', required=True,
              callback=validate_path,
              help='Output directory for MD simulation data')
@click.option('-i', '--sim_id', required=True, type=int,
              callback=validate_positive,
              help='Simulation ID in pipeline [0...N]')
@click.option('-t', '--topol', default=None, 
              callback=validate_path, help='Topology file')
@click.option('-c', '--chk', default=None,
              callback=validate_path,
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
def main(pdb, out, sim_id, topol, chk, length, report, gpu):
    openmm_simulate_amber_fs_pep(pdb,
                                 checkpnt=chk,
                                 GPU_index=gpu,
                                 checkpnt_fname=os.path.join(out, f'checkpnt-{sim_id}.chk'),
                                 output_traj=os.path.join(out, f'output-{sim_id}.dcd'),
                                 output_log=os.path.join(out, f'output-{sim_id}.log'),
                                 output_cm=os.path.join(out, f'output-cm-{sim_id}.h5'),
                                 report_time=report * u.picoseconds,
                                 sim_time=length * u.nanoseconds)

if __name__ == '__main__':
    main()

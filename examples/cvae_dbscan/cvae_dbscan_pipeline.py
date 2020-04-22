import os
import sys
import click
import shutil
from deepdrive import DeepDriveMD
from examples.cvae_dbscan.taskmanagers.md import MDTaskManager
from examples.cvae_dbscan.taskmanagers.preprocess import ContactMatrixTaskManager
from examples.cvae_dbscan.taskmanagers.ml import CVAETaskManager
from examples.cvae_dbscan.taskmanagers.outlier import OPTICSTaskManager


@click.command()
@click.option('-i', '--pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to initial pdb file')

def main(pdb_path):
    # Create directory structure to store pipeline data
    data_dir = os.path.join(os.getcwd(), 'data')
    for dir_name in ['md', 'preproc', 'ml', 'outlier', 'shared']:
        pipeline_dir = os.path.join(data_dir, dir_name, 'pipeline-0')
        if not os.path.exists(pipeline_dir):
            os.makedirs(pipeline_dir)
        else:
            raise Exception(f'data directory already exists. {pipeline_dir}')

    shared_pdb_dir = os.path.join(data_dir, 'shared', 'pipeline-0', 'pdb')
    os.makedirs(shared_pdb_dir)

    # Make 10 copies of pdb_path in shared directory for initial starting set
    for i in range(10):
        shutil.copyfile(src=pdb_path,
                        dst=os.path.join(shared_pdb_dir, f'input-{i}.pdb'))

    # Create a dictionary to describe four mandatory keys:
    # resource, walltime, cores and project.
    # Resource is 'local.localhost' to execute locally
    resources = {
        'resource': 'local.localhost', #'ornl.summit', #
        'queue': 'batch',       # 'killable'
        'schema': 'local',
        'walltime': 10,#60 * 2,     # 12
        'cpus': 1,#42 * 2,         # 20
        'gpus': 0,#6 * 2,          # 6*20
        'project': 'BIP179'
    }

    # Initialize hardware requirements and other parameters for each stage.
    # Note: each task_name must be unique.
    md_kwargs = {
        'num_sims': 1,#6*2,
        'sim_len': 10,
        'initial_sim_len': 10,
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 4,
            'thread_type': 'OpenMP'
        },
        # 'gpu_reqs': {
        #     'processes': 1,
        #     'process_type': None,
        #     'threads_per_process': 1,
        #     'thread_type': 'CUDA'
        # }
    }

    preproc_kwargs = {
        'cpu_reqs': {},
        'gpu_reqs': {}
    }

    ml_kwargs = {
        'num_ml' : 1,
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 4,
            'thread_type': 'OpenMP'
        },
        # 'gpu_reqs': {
        #     'processes': 1,
        #     'process_type': None,
        #     'threads_per_process': 1,
        #     'thread_type': 'CUDA'
        # }
    }

    outlier_kwargs = {
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 12,
            'thread_type': 'OpenMP'
        },
        # 'gpu_reqs': {
        #     'processes': 1,
        #     'process_type': None,
        #     'threads_per_process': 1,
        #     'thread_type': 'CUDA'
        # }
    }

    # Four lists must be created for each stage in the DeepDriveMD
    # pipeline. Each list contains at least one task manager responsible
    # for defining a set of tasks to run during each stage. Note, each
    # category can have multiple task manager objects defined.

    md_sims = [MDTaskManager(**md_kwargs)]
    preprocs = [ContactMatrixTaskManager(**preproc_kwargs)]
    ml_algs = [CVAETaskManager(**ml_kwargs)]
    outlier_algs = [OPTICSTaskManager(**outlier_kwargs)]

    # Initialize DeepDriveMD object to manage pipeline.
    cvae_optics_dd = DeepDriveMD(md_sims=md_sims,
                                 preprocs=preprocs,
                                 ml_algs=ml_algs,
                                 outlier_algs=outlier_algs,
                                 resources=resources)

    # Start running program on Summit.
    cvae_optics_dd.run()

if __name__ == '__main__':
    # Reroute error and standard output
    sys.stderr = open('./stderr.log', 'w')
    sys.stdout = open('./stdout.log', 'w')
    main()

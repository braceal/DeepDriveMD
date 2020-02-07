from deepdrive import DeepDriveMD
from .md import BasicMD
from .preprocess import ContactMatrix
from .ml import CVAE
from .outlier import DBSCAN


if __name__ == '__main__':

    # Create a dictionary to describe four mandatory keys:
    # resource, walltime, cores and project.
    # Resource is 'local.localhost' to execute locally
    resources = {
        'resource': 'ornl.summit',
        'queue': 'batch',       # 'killable'
        'schema': 'local',
        'walltime': 60 * 2,     # 12
        'cpus': 42 * 2,         # 20 
        'gpus': 6 * 2,          # 6*20
        'project': 'BIP179'
    }

    # Initialize hardware requirements and other parameters for each stage.
    # Note: each task_name must be unique.
    md_kwargs = {
        'num_sims': 6*2,
        'sim_len': 10,
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 4,
            'thread_type': 'OpenMP'
        },
        'gpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 1,
            'thread_type': 'CUDA'
        }
    }

    preproc_kwargs = {
        'cpu_reqs': {},
        'gpu_reqs': {}
    }

    ml_kwargs = {
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 4,
            'thread_type': 'OpenMP'
        },
        'gpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 1,
            'thread_type': 'CUDA'
        }
    }

    outlier_kwargs = {
        'cpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 12,
            'thread_type': 'OpenMP'
        },
        'gpu_reqs': { 
            'processes': 1,
            'process_type': None,
            'threads_per_process': 1,
            'thread_type': 'CUDA'
        }
    }

    # Four lists must be created for each stage in the DeepDriveMD
    # pipeline. Each list contains at least one task manager responsible
    # for defining a set of tasks to run during each stage. Note, each
    # category can have multiple task manager objects defined.

    md_sims = [BasicMD(**md_kwargs)]
    preprocs = [ContactMatrix(**preproc_kwargs)]
    ml_algs = [CVAE(**ml_kwargs)]
    outlier_algs = [DBSCAN(**outlier_kwargs)]

    # Initialize DeepDriveMD object to manage pipeline.
    cvae_dbscan_dd = DeepDriveMD(md_sims=md_sims,
                                 preprocs=preprocs,
                                 ml_algs=ml_algs,
                                 outlier_algs=outlier_algs,
                                 resources=resources)

    # Start running program on Summit.
    cvae_dbscan_dd.run()

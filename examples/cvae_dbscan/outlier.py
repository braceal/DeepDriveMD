import os
from radical.entk import Task
from deepdrive import TaskManager


class DBSCAN(TaskManager):
    def __init__(self, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        """
        super().__init__(cpu_reqs, gpu_reqs)

        self.cwd = os.getcwd()

    def tasks(self):
        """
        Returns
        -------
        set of tasks to be added to the outlier stage.

        """
        md_dir = f'{self.cwd}/data/md/pipeline-{pipeline_id}'
        cvae_dir = f'{self.cwd}/data/ml/pipeline-{pipeline_id}'
        shared_dir = f'{self.cwd}/data/shared/pipeline-{pipeline_id + 1}/pdb'
        outlier_dir = f'{self.cwd}/data/outlier/pipeline-{pipeline_id}'
        cm_data_path = f'{self.cwd}/data/preproc/pipeline-{pipeline_id}/cvae-input.h5'

        task = Task()

        # Specify modules for python and cuda, activate conda env.
        # Create output directory for generated files.
        task.pre_exec = ['module load python/3.7.0-anaconda3-5.3.0',
                         'module load cuda/9.1.85',
                         f'conda activate {self.cwd}/conda-env/',
                         f'mkdir -p {outlier_dir}',
                         f'mkdir -p {shared_path}']

        # Initialize eps dictionary that is shared and updated over
        # each round of the pipeline
        if pipeline_id == 0:
            task.pre_exec.append(f'touch {outlier_dir}/eps-{pipeline_id}.json')

        # Specify python outlier detection task
        task.executable = [f'{self.cwd}/conda-env/bin/python']
        task.arguments = [f'{self.cwd}/examples/cvae_dbscan/scripts/dbscan.py']

        # Arguments for outlier detection task
        task.arguments.extend(['--sim_path', md_dir,
                               '--shared_path', shared_dir,
                               '--cm_path', cm_data_path,
                               '--cvae_path', cvae_dir])

        # Specify hardware requirements
        task.cpu_reqs = self.cpu_reqs
        task.gpu_reqs = self.gpu_reqs

        return set(task)

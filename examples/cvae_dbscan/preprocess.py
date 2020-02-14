import os
from radical.entk import Task
from deepdrive import TaskManager


class ContactMatrix(TaskManager):
    def __init__(self, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        Note: both cpu_reqs, gpu_reqs are empty.

        """
        super().__init__(cpu_reqs, gpu_reqs)

        self.cwd = os.getcwd()

    def tasks(self, pipeline_id):
        """
        Returns
        -------
        Set of tasks to be added to the preprocessing stage.

        """
        md_dir = f'{self.cwd}/data/md/pipeline-{pipeline_id}'
        preproc_dir = f'{self.cwd}/data/preproc/pipeline-{pipeline_id}'
        
        task = Task()

        # Specify modules for python and cuda, activate conda env.
        # Create output directory for generated files.
        task.pre_exec = ['module load python/3.7.0-anaconda3-5.3.0',
                         f'conda activate {self.cwd}/conda-env/',
                         f'mkdir -p {preproc_dir}']

        # Specify python preprocessing task
        task.executable = [f'{self.cwd}/conda-env/bin/python']
        task.arguments = [f'{self.cwd}/examples/cvae_dbscan/scripts/contact_map.py']

        # Arguments for preprocessing task
        task.arguments.extend(['--sim_path', md_dir,
                               '--out', preproc_dir])

        return set(task)

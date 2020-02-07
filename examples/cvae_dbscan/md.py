import os
import time
from radical.entk import Task
from deepdrive import TaskManager


class BasicMD(TaskManager):
    def __init__(self, num_sims, sim_len, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        num_sims : int
            number of MD simulations to run

        sim_len : int
            Time (ns) to run MD simulations for

        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        """
        super().__init__(cpu_reqs, gpu_reqs)

        self.num_sims = num_sims
        self.sim_len = sim_len
        self.cwd = os.getcwd()


    def task(self, pipeline_id, sim_num, time_stamp):

        # TODO: update cuda version
        # TODO: next_pdb


        md_dir = f'{self.cwd}/data/md/pipeline-{pipeline_id}'
            
        task = Task()

        # Specify modules for python and cuda, activate conda env.
        # Create output directory for generated files.
        task.pre_exec = ['module load python/3.7.0-anaconda3-5.3.0',
                         'module load cuda/9.1.85',
                         f'conda activate {self.cwd}/conda-env/',
                         f'mkdir -p {md_dir}']

        # Specify python MD task
        task.executable = [f'{self.cwd}/conda-env/bin/python']
        task.arguments = [f'{self.cwd}/examples/cvae_dbscan/scripts/md.py']

        # Arguments for MD task
        task.arguments.extend(['--pdb_file', next_pdb(),
                               '--out', md_dir,
                               '--sim_id', f'{sim_num}',
                               '--len', sim_len,
                               '--sim_id', sim_num])

        # Assign hardware requirements
        task.cpu_reqs = self.cpu_reqs
        task.gpu_reqs = self.gpu_reqs

        return task


    def tasks(self, pipeline_id):
        """
        Returns
        -------
        set of tasks to be added to the MD stage

        """
        # TODO: incorporate or remove timestamp
        time_stamp = int(time.time())
        return {self.task(i, time_stamp, pipeline_id) for i in range(self.num_sims)}

            
import os
import glob
import time
from radical.entk import Task
from deepdrive import TaskManager


class BasicMD(TaskManager):
    def __init__(self, num_sims, sim_len, initial_sim_len, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        num_sims : int
            number of MD simulations to run

        sim_len : int
            Time (ns) to run MD simulations for

        initial_sim_len : int
            Time (ns) to run initial MD simulation batch for

        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        """
        super().__init__(cpu_reqs, gpu_reqs)

        self.num_sims = num_sims
        self.sim_len = sim_len
        self.initial_sim_len = initial_sim_len
        self.cwd = os.getcwd()

    def _task(self, pipeline_id, sim_num, time_stamp, md_dir, shared_dir, incomming_pbds):

        # TODO: update cuda version

        pdb_file = os.path.join(md_dir, f'input-{sim_num}.pdb')
            
        task = Task()

        # Specify modules for python and cuda, activate conda env.
        # Create output directory for generated files.
        task.pre_exec = ['module load python/3.7.0-anaconda3-5.3.0',
                         'module load cuda/9.1.85',
                         '. /sw/summit/python/3.7/anaconda3/5.3.0/etc/profile.d/conda.sh', 
                         f'conda activate {self.cwd}/conda-env/',
                         f'mkdir -p {md_dir}',
                         f'cp {incomming_pbds[sim_num]} {pdb_file}']

        # Specify python MD task
        task.executable = [f'{self.cwd}/conda-env/bin/python']
        task.arguments = [f'{self.cwd}/examples/cvae_dbscan/scripts/md.py']

        # Arguments for MD task
        task.arguments.extend(['--pdb', pdb_file,
                               '--out', md_dir,
                               '--sim_id', str(sim_num),
                               '--len', str(self.sim_len if pipeline_id else self.initial_sim_len)])

        # Specify hardware requirements
        task.cpu_reqs = self.cpu_reqs
        task.gpu_reqs = self.gpu_reqs

        return task


    def tasks(self, pipeline_id):
        """
        Returns
        -------
        set of tasks to be added to the MD stage.

        """

        md_dir = f'{self.cwd}/data/md/pipeline-{pipeline_id}'
        shared_dir = f'{self.cwd}/data/shared/pipeline-{pipeline_id}/pdb'
        incomming_pbds = glob.glob(os.path.join(shared_dir, '*.pdb'))

        if not incomming_pbds:
            print('No more PDB files to seed MD simulations')
            exit()

        # If there are fewer than self.num_sims incomming pbds then use all available
        num_sims = min(self.num_sims, len(incomming_pbds))

        # TODO: incorporate or remove timestamp
        time_stamp = int(time.time())
        return {self._task(pipeline_id, sim_num, time_stamp, md_dir, shared_dir, incomming_pbds)
                for sim_num in range(num_sims)}

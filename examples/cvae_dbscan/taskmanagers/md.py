import os
import glob
import time
from radical.entk import Task
from deepdrive import TaskManager


class MDTaskManager(TaskManager):
    def __init__(self, num_sims, sim_len, initial_sim_len,
                 cpu_reqs={}, gpu_reqs={}, prefix=os.getcwd()):
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

        prefix : str
            path from root to /DeepDriveMD directory

        """
        super().__init__(cpu_reqs, gpu_reqs, prefix)

        self.num_sims = num_sims
        self.sim_len = sim_len
        self.initial_sim_len = initial_sim_len

    def _task(self, pipeline_id, sim_num, time_stamp, md_dir, shared_dir, incomming_pbds):

        pdb_file = os.path.join(md_dir, f'input-{sim_num}.pdb')
            
        task = Task()

        self.load_environment(task)
        self.set_python_executable(task)
        self.assign_hardware(task)

        # Create output directory for generated files.
        task.pre_exec.extend([f'mkdir -p {md_dir}',
                              f'cp {incomming_pbds[sim_num]} {pdb_file}'])

        # Specify python MD task with arguments
        task.arguments = [f'{self.prefix}/examples/cvae_dbscan/scripts/md.py',
                          '--pdb', pdb_file,
                          '--out', md_dir,
                          '--sim_id', str(sim_num),
                          '--len', str(self.sim_len if pipeline_id else self.initial_sim_len)]

        return task

    def tasks(self, pipeline_id):
        """
        Returns
        -------
        set of tasks to be added to the MD stage.

        """

        md_dir = f'{self.prefix}/data/md/pipeline-{pipeline_id}'
        shared_dir = f'{self.prefix}/data/shared/pipeline-{pipeline_id}/pdb'
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

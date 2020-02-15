import os
import time
from deepdrive import TaskManager


class CVAE(TaskManager):
    def __init__(self, num_ml, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        num_ml : int
            number of ml models to train

        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        """
        super().__init__(cpu_reqs, gpu_reqs)

        self.num_ml = num_ml
        self.cwd = os.getcwd()


    def _task(self, pipeline_id, model_id, time_stamp):

        # Specify training hyperparameters
        # Select latent dimension for CVAE [3, ... self.num_ml]
        latent_dim = 3 + model_id
        epochs = 100
        batch_size = 512

        cvae_dir = f'{self.cwd}/data/ml/pipeline-{pipeline_id}'
        cm_data_path = f'{self.cwd}/data/preproc/pipeline-{pipeline_id}/cvae-input.h5'

        task = Task()

        # Specify modules for python and cuda, activate conda env.
        # Create output directory for generated files.
        task.pre_exec ['module load python/3.7.0-anaconda3-5.3.0',
                       'module load cuda/9.1.85',
                       '. /sw/summit/python/3.7/anaconda3/5.3.0/etc/profile.d/conda.sh',
                       f'conda activate {self.cwd}/conda-env/',
                       f'mkdir -p {cvae_dir}']

        # Specify python ML task
        task.executable = [f'{self.cwd}/conda-env/bin/python']
        task.arguments = [f'{self.cwd}/examples/cvae_dbscan/scripts/cvae.py']

        # Arguments for ML task
        task.arguments.extend(['--input', cm_data_path,
                               '--out', cvae_dir,
                               '--model_id', f'{model_id}',
                               '--epochs', f'{epochs}',
                               '--batch_size', f'{batch_size}',
                               '--latent_dim', f'{latent_dim}'])
        
        # Specify hardware requirements
        task.cpu_reqs = self.cpu_reqs
        task.gpu_reqs = self.gpu_reqs

        return task


    def tasks(self, pipeline_id):
        """
        Returns
        -------
        set of tasks to be added to the ML stage.

        """
        # TODO: incorporate or remove timestamp
        time_stamp = int(time.time())
        return {self._task(pipeline_id, i, time_stamp) for i in range(self.num_ml)}
            

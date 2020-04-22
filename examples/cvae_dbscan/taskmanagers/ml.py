import os
import time
from radical.entk import Task
from deepdrive import TaskManager


class CVAETaskManager(TaskManager):
    def __init__(self, num_ml, cpu_reqs={}, gpu_reqs={}, prefix=os.getcwd()):
        """
        Parameters
        ----------
        num_ml : int
            number of ml models to train

        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        prefix : str
            path from root to /DeepDriveMD directory

        """
        super().__init__(cpu_reqs, gpu_reqs, prefix)

        self.num_ml = num_ml


    def _task(self, pipeline_id, model_id, time_stamp):

        # Specify training hyperparameters
        # Select latent dimension for CVAE [3, ... self.num_ml]
        latent_dim = 3 + model_id
        epochs = 100
        batch_size = 512

        cvae_dir = f'{self.prefix}/data/ml/pipeline-{pipeline_id}'
        cm_data_path = f'{self.prefix}/data/preproc/pipeline-{pipeline_id}/cvae-input.h5'

        task = Task()

        self.load_environment(task)
        self.set_python_executable(task)
        self.assign_hardware(task)

        # Create output directory for generated files.
        task.pre_exec.extend([f'mkdir -p {cvae_dir}'])

        # Specify python ML task with arguments
        task.arguments = [f'{self.prefix}/examples/cvae_dbscan/scripts/cvae.py',
                          '--input', cm_data_path,
                          '--out', cvae_dir,
                          '--model_id', f'{model_id}',
                          '--epochs', f'{epochs}',
                          '--batch_size', f'{batch_size}',
                          '--latent_dim', f'{latent_dim}']
        
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

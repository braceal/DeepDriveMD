import os
from radical.entk import Task
from deepdrive import TaskManager


class ContactMatrixTaskManager(TaskManager):
    def __init__(self, cpu_reqs={}, gpu_reqs={}, prefix=os.getcwd()):
        """
        Parameters
        ----------
        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        prefix : str
            path from root to /DeepDriveMD directory

        Note: both cpu_reqs, gpu_reqs are empty.

        """
        super().__init__(cpu_reqs, gpu_reqs, prefix)

    def tasks(self, pipeline_id):
        """
        Returns
        -------
        Set of tasks to be added to the preprocessing stage.

        """
        md_dir = f'{self.prefix}/data/md/pipeline-{pipeline_id}'
        preproc_dir = f'{self.prefix}/data/preproc/pipeline-{pipeline_id}'
        
        task = Task()

        self.load_environment(task)
        self.set_python_executable(task)
        self.assign_hardware(task)

        # Create output directory for generated files.
        task.pre_exec.extend([f'mkdir -p {preproc_dir}'])

        # Specify python preprocessing task with arguments
        task.arguments = [f'{self.prefix}/examples/cvae_dbscan/scripts/contact_map.py',
                          '--sim_path', md_dir,
                          '--out', preproc_dir]

        return {task}

import os
from radical.entk import Task
from deepdrive import TaskManager


class OPTICSTaskManager(TaskManager):
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

        """
        super().__init__(cpu_reqs, gpu_reqs, prefix)

    def tasks(self, pipeline_id):
        """
        Returns
        -------
        set of tasks to be added to the outlier stage.

        """
        md_dir = f'{self.prefix}/data/md/pipeline-{pipeline_id}'
        cvae_dir = f'{self.prefix}/data/ml/pipeline-{pipeline_id}'
        shared_path = f'{self.prefix}/data/shared/pipeline-{pipeline_id + 1}/pdb'
        outlier_dir = f'{self.prefix}/data/outlier/pipeline-{pipeline_id}'
        cm_data_path = f'{self.prefix}/data/preproc/pipeline-{pipeline_id}/cvae-input.h5'

        task = Task()
        self.load_environment(task)
        self.set_python_executable(task)
        self.assign_hardware(task)

        # Create output directory for generated files.
        task.pre_exec.extend([f'mkdir -p {outlier_dir}',
                              f'mkdir -p {shared_path}'])

        # Initialize eps dictionary that is shared and updated over
        # each round of the pipeline
        if pipeline_id == 0:
            task.pre_exec.append(f'touch {outlier_dir}/eps-{pipeline_id}.json')

        # Specify python outlier detection task with arguments
        task.arguments = [f'{self.prefix}/examples/cvae_dbscan/scripts/dbscan.py',
                          '--sim_path', md_dir,
                          '--shared_path', shared_path,
                          '--cm_path', cm_data_path,
                          '--cvae_path', cvae_dir]

        return {task}

from abc import ABCMeta, abstractmethod

class TaskManager(metaclass=ABCMeta):
    def __init__(self, cpu_reqs, gpu_reqs, prefix):
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
        self.cpu_reqs = cpu_reqs
        self.gpu_reqs = gpu_reqs
        self.prefix = prefix

    def assign_hardware(self, task):
        """
        Assign GPU and CPU resources to task.

        Parameters
        ----------
        task : radical.entk.Task
        """
        if self.cpu_reqs:
            task.cpu_reqs = self.cpu_reqs
        if self.gpu_reqs:
            task.gpu_reqs = self.gpu_reqs

    def load_environment(self, task):
        """
        Loads modules and activates conda environment for
        DeepDriveMD script dependencies.

        Parameters
        ----------
        task : radical.entk.Task
        """
        task.pre_exec.extend(['module load python/3.6.6-anaconda3-5.3.0',
                              'module load hdf5/1.10.3',
                              'module load cuda/10.1.168',
                              '. /sw/summit/python/3.6/anaconda3/5.3.0/etc/profile.d/conda.sh',
                              f'conda activate {self.prefix}/conda-env/'])

    def set_python_executable(self, task):
        """
        Set task executable to python.

        Parameters
        ----------
        task : radical.entk.Task
        """
        task.executable = [f'{self.prefix}/conda-env/bin/python']

    @abstractmethod
    def tasks(self, pipeline_id):
        """
        Returns
        -------
        Set of tasks to be added to the stage.

        """
        raise NotImplementedError('Should implement tasks()')

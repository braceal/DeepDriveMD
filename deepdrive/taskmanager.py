from abc import ABCMeta, abstractmethod

class TaskManager(metaclass=ABCMeta):
    def __init__(self, cpu_reqs, gpu_reqs):
        """
        Parameters
        ----------
        cpu_reqs : dict
            contains cpu hardware requirments for task

        gpu_reqs : dict
            contains gpu hardware requirments for task

        """
        self.cpu_reqs = cpu_reqs
        self.gpu_reqs = gpu_reqs

    @abstractmethod
    def tasks(self, pipeline_id):
        """
        Returns
        -------
        Set of tasks to be added to the stage.

        """
        raise NotImplementedError('Should implement tasks()')

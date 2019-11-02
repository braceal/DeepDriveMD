from abc import ABCMeta, abstractmethod

class TaskMan(metaclass=ABCMeta):
	def __init__(self, task_name, cpu_reqs, gpu_reqs):
		"""
		Parameters
		----------
		task_name : str
			name of particular task

		cpu_reqs : dict
			contains cpu hardware requirments for task

		gpu_reqs : dict
			contains gpu hardware requirments for task

		"""
		self.task_name = task_name
		self.cpu_reqs = cpu_reqs
		self.gpu_reqs = gpu_reqs
		self.input = dict()


	@abstractmethod
	def output(self):
		"""
		Effects
		-------
		Defines a dictionary of output to be passed to the next 
		stages TaskMan's.

		"""
		raise NotImplementedError('Should implement output()')


	@abstractmethod
	def tasks(self):
		"""
		Returns
		-------
		Set of tasks to be added to the stage.

		"""
		raise NotImplementedError('Should implement tasks()')


	def subscribe(self, taskmans):
		"""
		Parameters
		----------
		taskmans : list
			list of DeepDriveMD.taskman.TaskMan objects 
			which manage different types of tasks within the pipeline.
				
		Effects
		-------
		Sets input to the output of dependencies.

		"""

		task_names = set(t.task_name for t in taskmans)

		if self.task_name in task_names:
			raise Exception('Cannot subscribe to self')

		if len(task_names) != len(taskmans):
			raise Exception('Naming collision between task managers')

		self.input = dict((t.task_name, t.output()) for t in taskmans)

import os
from radical.entk import Task
from DeepDriveMD.taskman import TaskMan


class DBSCAN(TaskMan):
	def __init__(self, task_name, num_ml, cpu_reqs, gpu_reqs):
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
		super().__init__(task_name, cpu_reqs, gpu_reqs)
	
		self.conda_path = '/ccs/home/hm0/.conda/envs/omm'
		self.cwd = os.getcwd()


	def output(self):
		"""
		Effects
		-------
		Defines a dictionary of output to be passed to 
		other subscribing tasks.

		Returns
		-------
		output dictionary

		"""
        return {'outlier_filepath': '%s/Outlier_search/restart_points.json' % self.cwd}


	def tasks(self):
		"""
		Returns
		-------
		set of tasks to be added to the MD stage

		"""
		task = Task() 
		task.pre_exec = ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh',
						 'module load cuda/9.1.85',
						 'conda activate %s' % self.conda_path, 
						 'export PYTHONPATH=%s/CVAE_exps:$PYTHONPATH' % self.cwd, 
						 'cd %s/Outlier_search' % self.cwd] 
		task.executable = ['%s/bin/python' % self.conda_path] 
		task.arguments = ['outlier_locator.py', '--md', '../MD_exps/fs-pep', '--cvae', '../CVAE_exps',
		        		  '--pdb', '../MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb', 
		        		  '--ref', '../MD_exps/fs-pep/pdb/fs-peptide.pdb']

		task.cpu_reqs = self.cpu_reqs
		task.gpu_reqs = self.gpu_reqs

		return set(task)





			
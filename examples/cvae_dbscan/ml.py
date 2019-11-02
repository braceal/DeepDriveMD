import os
import time
from radical.entk import Task
from DeepDriveMD.taskman import TaskMan


class CVAE(TaskMan):
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
		self.num_ml = num_ml

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
        return {'--sim_path': 'base/MD_exps/fs-pep'}

	def task(self, ml_num, time_stamp):

		# Select latent dimension for CVAE
    	dim = ml_num + 3 
        cvae_dir = 'cvae_runs_%.2d_%d' % (dim, time_stamp + ml_num) 

        task = Task()

        task.pre_exec ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh',
      				   'module load cuda/9.1.85',
        			   'conda activate %s' % self.conda_path, 
					   'export PYTHONPATH=%s/CVAE_exps:$PYTHONPATH' % self.cwd
        			   'cd %s/CVAE_exps' % self.cwd,
      				   'mkdir -p {0} && cd {0}'.format(cvae_dir)]

        task.executable = ['%s/bin/python' % self.conda_path]

        task.arguments = ['%s/CVAE_exps/train_cvae.py' % self.cwd, 
                          '--h5_file', self.input['ContactMatrix']['--h5_file'], 
                          '--dim', dim] 
        
        task.cpu_reqs = self.cpu_reqs
        task.gpu_reqs = self.gpu_reqs

        return task


	def tasks(self):
		"""
		Returns
		-------
		set of tasks to be added to the MD stage

		"""
		time_stamp = int(time.time())
		return {self.task(i, time_stamp) for i in range(self.num_ml)}

			
import os
import time
from radical.entk import Task
from DeepDriveMD.taskman import TaskMan


class BasicMD(TaskMan):
	def __init__(self, task_name, num_sims, initial_len, 
				 iter_len, cpu_reqs, gpu_reqs):
		"""
		Parameters
		----------
		num_sims : int
			number of simulations to run

		initial_len : int
            Time (ns) to run initial simulation

        iter_len : int
            Time (ns) to run iterative simulation

		cpu_reqs : dict
			contains cpu hardware requirments for task

		gpu_reqs : dict
			contains gpu hardware requirments for task

		"""
		super().__init__(task_name, cpu_reqs, gpu_reqs)
		self.num_sims = num_sims
		self.initial_len = initial_len
		self.iter_len = iter_len

		self.conda_path = '/ccs/home/hm0/.conda/envs/omm'
		self.cwd = os.getcwd()

		self.initial_MD = True

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
        return {'--sim_path': '%s/MD_exps/fs-pep' % self.cwd}

	def task(self, sim_num, time_stamp):

		outlier_filepath = self.input['DBSCAN']['outlier_filepath']

    	if os.path.exists(outlier_filepath): 
	        self.initial_MD = False 
	        with open(outlier_filepath, 'r') as f:
	        	outlier_list = json.load(f) 
	        
		task = Task()
		task.pre_exec = ['. /sw/summit/python/2.7/anaconda2/5.3.0/etc/profile.d/conda.sh',
						 'module load cuda/9.1.85'
						 'conda activate %s' % self.conda_path,
						 'export PYTHONPATH=%s/MD_exps:$PYTHONPATH' % self.cwd,
						 'cd %s/MD_exps/fs-pep' % self.cwd,
						 'mkdir -p omm_runs_%d && cd omm_runs_%d' % (time_stamp+sim_num, time_stamp+sim_num)]

		task.executable = ['%s/bin/python' % self.conda_path]
		task.arguments = ['%s/MD_exps/fs-pep/run_openmm.py' % self.cwd]

        # Pick initial point of simulation 
        task.arguments.append('--pdb_file')

        if self.initial_MD or sim_num >= len(outlier_list): 
            task.arguments.append('%s/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb' % self.cwd)

        elif outlier_list[sim_num].endswith('pdb'): 
            task.arguments.append(outlier_list[sim_num])
            task.pre_exec.append('cp %s ./' % outlier_list[sim_num]) 

        elif outlier_list[sim_num].endswith('chk'): 
            task.arguments.extend(['%s/MD_exps/fs-pep/pdb/100-fs-peptide-400K.pdb' % self.cwd,
                    			  '-c', outlier_list[sim_num]]) 
            task.pre_exec.append('cp %s ./' % outlier_list[sim_num])

        # How long to run the simulation
        task.arguments.extend(['--length', 
        		self.initial_len if self.initial_MD else self.iter_len])

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
		return {self.task(i, time_stamp) for i in range(self.num_sims)}

			
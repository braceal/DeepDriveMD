from DeepDriveMD.deepdrive import DeepDriveMD
from .md import BasicMD
from .preprocess import ContactMatrix
from .ml import CVAE
from .outlier import DBSCAN



if __name__ == '__main__':

	# Create a dictionary to describe four mandatory keys:
	# resource, walltime, cores and project.
	# Resource is 'local.localhost' to execute locally
    resources = {
		'resource': 'ornl.summit',
		'queue': 'batch', 		# 'killable'
		'schema': 'local',
		'walltime': 60 * 2, 	# 12
		'cpus': 42 * 2, 		# 20 
		'gpus': 6 * 2,  		# 6*20
		'project': 'BIP179'
	}

	# Initialize hardware requirements and other parameters for each stage.
	# Note: each task_name must be unique.
	md_kwargs = {
		'task_name': 'BasicMD',
		'num_sims': 6*2,
		'initial_len': 10,
		'iter_len': 10,
		'cpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 4,
			'thread_type': 'OpenMP'
		},
		'gpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 1,
			'thread_type': 'CUDA'
		}
	}

	preproc_kwargs = {
		'task_name': 'ContactMatrix',
		'cpu_reqs': {},
		'gpu_reqs': {}
	}

	ml_kwargs = {
		'task_name': 'CVAE',
		'cpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 4,
			'thread_type': 'OpenMP'
		},
		'gpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 1,
			'thread_type': 'CUDA'
		}
	}

	outlier_kwargs = {
		'task_name': 'DBSCAN',
		'cpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 12,
			'thread_type': 'OpenMP'
		},
		'gpu_reqs': { 
			'processes': 1,
			'process_type': None,
			'threads_per_process': 1,
			'thread_type': 'CUDA'
		}
	}

	# Four lists must be created for each stage in the DeepDriveMD
	# pipeline. Each list contains atleast one task manager responsible
	# for defining a set of tasks to run during each stage. Note, each
	# category can have multiple task manager objects defined. 
	# Communication between task managers is handled through 
	# subscriptions (below).

	md_sims = [BasicMD(**md_kwargs)]
	preprocs = [ContactMatrix(**preproc_kwargs)]
	ml_algs = [CVAE(**ml_kwargs)]
	outlier_algs = [DBSCAN(**outlier_kwargs)]


	# Subscribe task managers to output of other tasks.
	# In this case we only have one task in each stage.

	for md_sim in md_sims:
		md_sim.subscribe(outlier_algs)

	for preproc in preprocs:
		preproc.subscribe(md_sims)

	for ml_alg in ml_algs:
		ml_alg.subscribe(preprocs)

	for outlier_alg in outlier_algs:
		outlier_alg.subscribe(ml_algs)


	# Initialize DeepDriveMD object to manage pipeline.
	cvae_dbscan_dd = DeepDriveMD(md_sims=md_sims,
								 preprocs=preprocs,
								 ml_algs=ml_algs,
								 outlier_algs=outlier_algs,
								 resources=resources)


	# Start running program on Summit.
	cvae_dbscan_dd.run()

from __future__ import print_function
import os
from radical.entk import Pipeline, Stage, Task, AppManager


def construct_task(task_data):
    """
    Paramters
    ---------
    task_data : deepdrive.Task
        Stores task meta data to construct entk Task

    Returns
    -------
    task : Task
        entk Task

    """
    task = Task()
    task.pre_exec = task_data.pre_exec
    task.executable = task_data.executable
    task.arguments = task_data.arguments
    task.cpu_reqs = task_data.cpu_reqs
    task.gpu_reqs = task_data.gpu_reqs
    return task


class EntkDriver:
    """
    Implements an interface for the DeepDriveMD computational 
    motif presented in: https://arxiv.org/abs/1909.07817
    """
    def __init__(self, deepdrive):
        """
        Parameters
        ----------
        deepdrive : deepdrive.DeepDriveMD 
            Stores user pipeline details
        """
        self.deepdrive = deepdrive

        # TODO: Move outside of class? Maybe put in __init__.py
        #       or bash setup script
        # Set default verbosity
        if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
            os.environ['RADICAL_ENTK_REPORT'] = 'True'

        # Initialize pipeline
        self._pipeline = Pipeline()
        self._pipeline.name = self.deepdrive.pipeline_name

        # Sets pipeline stages
        self.pipeline()

        # Create Application Manager
        self.appman = AppManager(hostname=os.environ.get('RMQ_HOSTNAME'), 
                                 port=int(os.environ.get('RMQ_PORT')))
        self.appman.resource_desc = resources

        # Assign the workflow as a list of Pipelines to the Application Manager. In
        # this way, all the pipelines in the list will execute concurrently.
        self.appman.workflow = [self._pipeline]


    def run(self):
        """
        Effects
        -------
        Runs the Application Manager. 
        
        """
        self.appman.run()


    def generate_MD_stage(self):
        stage = Stage()
        stage.name = self.deepdrive.md_stage_name
        for sim in self.deepdrive.md_sims:
            stage.add_tasks(set(map(constuct_task, sim.tasks())))
        return stage


    def generate_preprocess_stage(self):
        stage = Stage()
        stage.name = self.deepdrive.pre_stage_name
        for preproc in self.deepdrive.preprocs:
            stage.add_tasks(set(map(constuct_task, preproc.tasks())))
        return stage


    def generate_ml_stage(self):
        stage = Stage()
        stage.name = self.deepdrive.ml_stage_name
        for alg in self.deepdrive.ml_algs:
            stage.add_tasks(set(map(constuct_task, alg.tasks())))
        return stage


    def generate_outlier_stage(self):
        stage = Stage()
        stage.name = self.deepdrive.outlier_stage_name
        for alg in self.deepdrive.outlier_algs:
            stage.add_tasks(set(map(constuct_task, alg.tasks())))

        stage.post_exec = {
            'condition': lambda: self.deepdrive.current_iter < self.deepdrive.max_iter,
            'on_true': self.pipeline,
            'on_false': lambda: print('Done')
        }

        return stage


    def pipeline(self):
        """
        Effects
        -------
        Adds stages to pipeline.

        """
        if self.deepdrive.current_iter:
            print('Finishing pipeline iteration {} of {}'.format(self.deepdrive.current_iter, 
                                                                 self.deepdrive.max_iter)) 
        # MD stage
        s1 = self.generate_md_stage()
        self._pipeline.add_stages(s1)

        # Preprocess stage
        s2 = self.generate_preprocess_stage() 
        self._pipeline.add_stages(s2)  

        # Learning stage
        s3 = self.generate_ml_stage()
        self._pipeline.add_stages(s3)

        # Outlier identification stage
        s4 = self.generate_outlier_stage(settings) 
        self._pipeline.add_stages(s4) 

        self.deepdrive.current_iter += 1

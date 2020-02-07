import os
from radical.entk import Pipeline, Stage, Task, AppManager


class DeepDriveMD:
    """
    Implements an interface for the DeepDriveMD computational 
    motif presented in: https://arxiv.org/abs/1909.07817
    """
    def __init__(self, md_sims, preprocs, ml_algs, outlier_algs, resources,
                 max_iter=1, pipeline_name='MD_ML', md_stage_name='MD',
                 pre_stage_name='Preprocess', ml_stage_name='ML',
                 outlier_stage_name='Outlier'):

        """
        Parameters
        ----------
        md_sims : list
            list of DeepDriveMD.taskmanager.TaskManager objects
            which manage simulations

        preprocs : list
            list of DeepDriveMD.taskmanager.TaskManager objects
            which manage data preprocessing

        ml_algs : list
            list of DeepDriveMD.taskmanager.TaskManager objects
            which manage representation learning
        
        outlier_algs : list
            list of DeepDriveMD.taskmanager.TaskManager objects
            which manage outlier detection

        resources : dict
            Configuration settings for running on Summit

        max_iter : int
            Max number of iterations through the pipeline
    
        pipeline_name : str
            Name of computational pipeline

        md_stage_name : str
            Name of MD stage

        pre_stage_name : str
            Name of preprocessing stage

        ml_stage_name : str
            Name of ML stage

        outlier_stage_name : str
            Name of outlier detection stage

        """

        # Number of iterations through the pipeline
        self.current_iter = 0
        self.max_iter = max_iter

        # TODO: Move outside of class? Maybe put in __init__.py
        #       or bash setup script
        # Set default verbosity
        if os.environ.get('RADICAL_ENTK_VERBOSE') is None:
            os.environ['RADICAL_ENTK_REPORT'] = 'True'

        # Initialize pipeline
        self.__pipeline = Pipeline()
        self.__pipeline.name = pipeline_name
        self.md_stage_name = md_stage_name
        self.pre_stage_name = pre_stage_name
        self.ml_stage_name = ml_stage_name
        self.outlier_stage_name = outlier_stage_name

        # Set stage task managers
        self.md_sims = md_sims
        self.preprocs = preprocs
        self.ml_algs = ml_algs
        self.outlier_algs = outlier_algs

        # Sets pipeline stages
        self.pipeline()

        # Create Application Manager
        self.appman = AppManager(hostname=os.environ.get('RMQ_HOSTNAME'), 
                                 port=int(os.environ.get('RMQ_PORT')))
        self.appman.resource_desc = resources

        # Assign the workflow as a list of Pipelines to the Application Manager. In
        # this way, all the pipelines in the list will execute concurrently.
        self.appman.workflow = [self.__pipeline]


    def run(self):
        """
        Effects
        -------
        Runs the Application Manager. 
        
        """
        self.appman.run()


    def generate_MD_stage(self):
        stage = Stage()
        stage.name = self.md_stage_name
        for sim in self.md_sims:
            stage.add_tasks(set(sim.tasks()))
        return stage


    def generate_preprocess_stage(self):
        stage = Stage()
        stage.name = self.pre_stage_name
        for preproc in self.preprocs:
            stage.add_tasks(set(preproc.tasks()))
        return stage


    def generate_ml_stage(self):
        stage = Stage()
        stage.name = self.ml_stage_name
        for alg in self.ml_algs:
            stage.add_tasks(set(alg.tasks()))
        return stage


    def generate_outlier_stage(self):
        stage = Stage()
        stage.name = self.outlier_stage_name
        for alg in self.outlier_algs:
            stage.add_tasks(set(alg.tasks()))

        stage.post_exec = {
            'condition': lambda: self.current_iter < self.max_iter,
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
        if self.current_iter:
            print('Finishing pipeline iteration {} of {}'.format(self.current_iter, 
                                                                 self.max_iter)) 
        # MD stage
        s1 = self.generate_md_stage()
        self.__pipeline.add_stages(s1)

        # Preprocess stage
        s2 = self.generate_preprocess_stage() 
        self.__pipeline.add_stages(s2)  

        # Learning stage
        s3 = self.generate_ml_stage()
        self.__pipeline.add_stages(s3)

        # Outlier identification stage
        s4 = self.generate_outlier_stage(settings) 
        self.__pipeline.add_stages(s4) 

        self.current_iter += 1

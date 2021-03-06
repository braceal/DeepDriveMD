import os
from collections import namedtuple
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
        # Checks environment variables are set
        self._validate_environment()

        # Number of iterations through the pipeline
        self.current_iter = 0
        self.max_iter = max_iter

        # Neatly stores stage name and taskmanagers
        StageData = namedtuple('StageData', ['name', 'taskmanagers'])

        # Dictionary storing name and taskmanagers for each stage
        self.stages = {'md': StageData(md_stage_name, md_sims),
                       'preprocess': StageData(pre_stage_name, preprocs),
                       'ml': StageData(ml_stage_name, ml_algs),
                       'outlier': StageData(outlier_stage_name, outlier_algs)}

        # Initialize pipeline
        self.__pipeline = Pipeline()
        self.__pipeline.name = pipeline_name

        # Sets pipeline stages
        self._pipeline()

        # Create Application Manager
        self.appman = AppManager(hostname=os.environ.get('RMQ_HOSTNAME'),
                                 port=int(os.environ.get('RMQ_PORT')))

        # Assign hardware resources to application
        self.appman.resource_desc = resources

        # Assign the workflow as a set of Pipelines to the Application Manager. In
        # this way, all the pipelines in the set will execute concurrently.
        self.appman.workflow = {self.__pipeline}

    def run(self):
        """
        Effects
        -------
        Runs the Application Manager. 
        
        """
        self.appman.run()

    def _validate_environment(self):
        if not os.environ.get('RADICAL_ENTK_VERBOSE'):
            os.environ['RADICAL_ENTK_REPORT'] = 'True'

        # All envars must be set prior
        envars = ['RMQ_HOSTNAME', 'RMQ_PORT', 'RADICAL_PILOT_DBURL',
                  'RADICAL_PILOT_PROFILE', 'RADICAL_ENTK_PROFILE']

        for envar in envars:
            if not os.environ.get(envar):
                raise Exception(f'{envar} environment variable not set')
        
    def _generate_stage(self, stage_type):
        """
        Parameters
        ----------
        stage_type : str
            key into self.stages dictionary to retrieve stage name and taskmanagers.
        """
        stage = Stage()
        stage.name = self.stages[stage_type].name
        for taskman in self.stages[stage_type].taskmanagers:
            stage.add_tasks(set(taskman.tasks(self.current_iter)))
        return stage

    def _condition(self):
        if self.current_iter < self.max_iter:
            self._pipeline()
        print('Done')

    def _pipeline(self):
        """
        Effects
        -------
        Adds stages to pipeline.

        """
        if self.current_iter:
            print(f'Finished pipeline iteration {self.current_iter} of {self.max_iter}')

        # Add the first three stages to the pipeline
        for stage_type in ['md', 'preprocess', 'ml']:
            self.__pipeline.add_stages(self._generate_stage(stage_type))

        # Generate last stage seperate to add post execution step
        last_stage = self._generate_stage('outlier')

        # Set post execution for last stage
        last_stage.post_exec = self._condition

        self.current_iter += 1

        self.__pipeline.add_stages(last_stage)

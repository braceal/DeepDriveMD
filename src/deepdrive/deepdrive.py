import os
import Pyro4
import threading
from subprocess import Popen, PIPE

@Pyro4.expose
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
            list of DeepDriveMD.taskman.TaskMan objects 
            which manage simulations

        preprocs : list
            list of DeepDriveMD.taskman.TaskMan objects 
            which manage data preprocessing

        ml_algs : list
            list of DeepDriveMD.taskman.TaskMan objects 
            which manage representation learning
        
        outlier_algs : list
            list of DeepDriveMD.taskman.TaskMan objects 
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

        self.pipeline = pipeline_name
        self.md_stage_name = md_stage_name
        self.pre_stage_name = pre_stage_name
        self.ml_stage_name = ml_stage_name
        self.outlier_stage_name = outlier_stage_name

        # Set stage task managers
        self.md_sims = md_sims
        self.preprocs = preprocs
        self.ml_algs = ml_algs
        self.outlier_algs = outlier_algs


def serve(daemon):
    """
    Effects
    -------
    Starts blocking requestloop. Waits on requests from
    process running entk pipeline.

    """
    daemon.requestLoop()


def run(dd):
    """
    Parameters
    ----------
    dd : DeepDriveMD
        contains all metadata to run the DeepDriveMD pipeline

    Effects
    -------
    Starts entk driver for DeepDriveMD pipeline.
    
    """
    # Initialize daemon
    daemon = Pyro4.Daemon()
    
    # Register DeepDriveMD object 
    uri = daemon.register(dd)
    
    # Start server on this process in another thread
    threading.Thread(target=serve, args=(daemon,)).start()

    # Start new process to run entk backend. The main function
    # retrieves the uri corresponding to the DeepDriveMD object
    # and allows the ability to communicate between python2 and python3.
    # TODO: write bash script to enter python2 virtual env
    process = Popen(['python2', 
                     '{}/entkdriver/__main__.py'.format(os.path.abspath('.')), 
                     '--uri', str(uri)], 
                     stdout=PIPE, stderr=PIPE)

    stdout, stderr = process.communicate()
    result = stdout.decode('utf-8').split('\n')
    print(result)

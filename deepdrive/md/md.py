import random
import parmed as pmd
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
import molecules.sim as sim



# TODO: use molecules package instead

def openmm_simulate_amber_fs_pep(pdb_file, top_file=None, checkpnt_fname='checkpnt.chk', 
                                 checkpnt=None, GPU_index=0,
                                 output_traj='output.dcd', output_log='output.log', output_cm=None,
                                 report_time=10*u.picoseconds,sim_time=10*u.nanoseconds, 
                                 platform='CUDA'):
    """
    Start and run an OpenMM NVT simulation with Langevin integrator at 2 fs 
    time step and 300 K. The cutoff distance for nonbonded interactions were 
    set at 1.2 nm and LJ switch distance at 1.0 nm, which commonly used with
    Charmm force field. Long-range nonbonded interactions were handled with PME.  

    Parameters
    ----------
    pdb_file : coordinates file (.gro, .pdb, ...)
        This is the molecule configuration file contains all the atom position
        and PBC (periodic boundary condition) box in the system. 
   
    checkpnt : None or check point file to load 
        
    GPU_index : Int or Str 
        The device # of GPU to use for running the simulation. Use Strings, '0,1'
        for example, to use more than 1 GPU
  
    output_traj : the trajectory file (.dcd)
        This is the file stores all the coordinates information of the MD 
        simulation results. 
  
    output_log : the log file (.log) 
        This file stores the MD simulation status, such as steps, time, potential
        energy, temperature, speed, etc.
 
    output_cm : the h5 file contains contact map information

    report_time : 10 ps
        The program writes its information to the output every 10 ps by default 

    sim_time : 10 ns
        The timespan of the simulation trajectory

    platform : str
        Name of platform. Options: 'CUDA', 'OpenCL', or 'CPU'

    """

    if top_file: 
        pdb = pmd.load_file(top_file, xyz=pdb_file)
        system = pdb.createSystem(nonbondedMethod=app.CutoffNonPeriodic, 
                nonbondedCutoff=1.0*u.nanometer, constraints=app.HBonds, 
                implicitSolvent=app.OBC1)
    else: 
        pdb = pmd.load_file(pdb_file)
        forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, 
                nonbondedCutoff=1.0*u.nanometer, constraints=app.HBonds)

    dt = 0.002*u.picoseconds
    integrator = omm.LangevinIntegrator(300*u.kelvin, 91.0/u.picosecond, dt)
    integrator.setConstraintTolerance(0.00001)

    # Select platform
    if platform is 'CUDA':
        platform = omm.Platform_getPlatformByName('CUDA')
        properties = {'DeviceIndex': str(GPU_index), 'CudaPrecision': 'mixed'}
    elif platform is 'OpenCL':
        platform = omm.Platform_getPlatformByName('OpenCL')
        properties = {'DeviceIndex': str(GPU_index)}
    elif platform is 'CPU':
        platform, properties = None, None
    else:
        raise ValueError(f'Invalid platform name: {platform}')

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setPositions(random.choice(pdb.get_coordinates())/10) #parmed \AA to OpenMM nm

    # equilibrate
    simulation.minimizeEnergy() 
    simulation.context.setVelocitiesToTemperature(300*u.kelvin, random.randint(1, 10000))
    simulation.step(int(100*u.picoseconds / (2*u.femtoseconds)))

    report_freq = int(report_time/dt)
    simulation.reporters.append(app.DCDReporter(output_traj, report_freq))
    if output_cm:
        simulation.reporters.append(sim.ContactMapReporter(output_cm, report_freq))

    simulation.reporters.append(app.StateDataReporter(output_log,
            report_freq, step=True, time=True, speed=True,
            potentialEnergy=True, temperature=True, totalEnergy=True))
    simulation.reporters.append(app.CheckpointReporter(checkpnt_fname, report_freq))

    if checkpnt:
        simulation.loadCheckpoint(checkpnt)
    nsteps = int(sim_time/dt)
    simulation.step(nsteps)

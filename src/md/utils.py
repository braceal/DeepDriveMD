import random
import parmed as pmd
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app
from molecules.utils import triu_to_full
from openmm_reporter import ContactMapReporter

# TODO: use molecules package instead

def openmm_simulate_amber_fs_pep(pdb_file, top_file=None, check_point=None, GPU_index=0,
                                 output_traj="output.dcd", output_log="output.log", output_cm=None,
                                 report_time=10*u.picoseconds, sim_time=10*u.nanoseconds):
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
   
    check_point : None or check point file to load 
        
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
    """

    if top_file: 
        pdb = pmd.load_file(top_file, xyz = pdb_file)
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

    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        properties = {'DeviceIndex': str(GPU_index), 'CudaPrecision': 'mixed'}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        properties = {'DeviceIndex': str(GPU_index)}

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

    simulation.context.setPositions(random.choice(pdb.get_coordinates())/10) #parmed \AA to OpenMM nm

    # equilibrate
    simulation.minimizeEnergy() 
    simulation.context.setVelocitiesToTemperature(300*u.kelvin, random.randint(1, 10000))
    simulation.step(int(100*u.picoseconds / (2*u.femtoseconds)))

    report_freq = int(report_time/dt)
    simulation.reporters.append(app.DCDReporter(output_traj, report_freq))
    if output_cm:
        simulation.reporters.append(ContactMapReporter(output_cm, report_freq))
    simulation.reporters.append(app.StateDataReporter(output_log,
            report_freq, step=True, time=True, speed=True,
            potentialEnergy=True, temperature=True, totalEnergy=True))
    simulation.reporters.append(app.CheckpointReporter('checkpnt.chk', report_freq))

    if check_point:
        simulation.loadCheckpoint(check_point)
    nsteps = int(sim_time/dt)
    simulation.step(nsteps)


def open_h5(h5_file):
    """
    Opens file in single write multiple reader mode
    libver specifies newest available version,
    may not be backwards compatable

    Parameters
    ----------
    h5_file : str
        name of h5 file to open

    Returns
    -------
    open h5py file

    """
    return h5py.File(h5_file, 'r', libver='latest', swmr=True)


def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae.

    Parameters
    ----------
    cm_data_lists : list
        list containing h5py dataset objects of contact matrices

    """
    cm_all = np.hstack(cm_data_lists)

    # Transfer upper triangle to full matrix
    cm_data_full = np.array(list(map(triu_to_full, cm_all.T)))

    # Padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x % 2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # Reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input

"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 merge.py checkpoint    
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

The simulation should take under 5 process-minutes to run.

Usage:
    rayleigh_benard.py [--Rayleigh=<Rayleigh> --Prandtl=<Prandtl> --restart=<restart_file>] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --restart=<restart_file>   Restart from checkpoint
    
"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras.checkpointing import Checkpoint

import logging
logger = logging.getLogger(__name__)

def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, restart=None):
    # input parameters
    logger.info("Ra = {}, Pr = {}".format(Rayleigh, Prandtl))
    
    # Parameters
    Lx, Lz = (4., 1.)

    # Create bases and domain
    x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
    problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
    problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
    problem.parameters['F'] = F = 1
    problem.add_equation("dx(u) + wz = 0")
    problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz))             = -(u*dx(b) + w*bz)")
    problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
    problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
    problem.add_equation("bz - dz(b) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")
    problem.add_bc("left(b) = left(-F*z)")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(b) = right(-F*z)")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("integ(p, 'z') = 0", condition="(nx == 0)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')

    # Checkpointing
    checkpoint = Checkpoint('./')
    checkpoint.set_checkpoint(solver, wall_dt=60)
    
    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    b = solver.state['b']
    bz = solver.state['bz']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    if restart is None:
        # Linear background + perturbations damped at walls
        zb, zt = z_basis.interval
        pert =  1e-3 * noise * (zt - z) * (z - zb)
        b['g'] = -F*(z - pert)
        b.differentiate('z', out=bz)
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        
    # Integration parameters
    solver.stop_sim_time = 30
    solver.stop_wall_time = 30 * 60.
    solver.stop_iteration = np.inf

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=50)
    snapshots.add_task("p")
    snapshots.add_task("b")
    snapshots.add_task("u")
    snapshots.add_task("w")

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=10, safety=1,
                         max_change=1.5, min_change=0.5, max_dt=0.1)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
    flow.add_property("sqrt(u*u + w*w) / R", name='Re')




    # Main loop
    try:
        logger.info('Starting loop')
        start_time = time.time()
        while solver.ok:
            dt = CFL.compute_dt()
            solver.step(dt)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %f' %flow.max('Re'))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

        
if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__) 
    Rayleigh_Benard(Rayleigh=float(args['--Rayleigh']),
                    Prandtl=float(args['--Prandtl']),
                    restart=(args['--restart']))


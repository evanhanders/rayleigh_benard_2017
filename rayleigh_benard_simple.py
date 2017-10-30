"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

Usage:
    rayleigh_benard.py [options] 

Options:
    --Rayleigh=<Rayleigh>      Rayleigh number [default: 1e6]
    --Prandtl=<Prandtl>        Prandtl number = nu/kappa [default: 1]
    --nz=<nz>                  Vertical resolution [default: 128]
    --nx=<nx>                  Horizontal resolution; if not set, nx=aspect*nz_cz
    --aspect=<aspect>          Aspect ratio of problem [default: 4]
    --viscous_heating          Include viscous heating

    --fixed_flux               Fixed flux boundary conditions top/bottom
    --mixed_flux_T             Fixed flux (bot) and fixed temp (top) bcs
    --fixed_T                  Fixed temperature boundary conditions top/bottom; default if no choice is made

    --stress_free              Stress free boundary conditions top/bottom
    --no_slip                  no slip boundary conditions top/bottom; default if no choice is made
    
    --run_time=<run_time>             Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_bouy>   Run time, in buoyancy times
    --run_time_iter=<run_time_iter>   Run time, number of iterations; if not set, n_iter=np.inf

    --restart=<restart_file>   Restart from checkpoint

    --max_writes=<max_writes>              Writes per file for files other than slices and coeffs [default: 20]
    --max_slice_writes=<max_slice_writes>  Writes per file for slices and coeffs [default: 20]
    
    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --no_coeffs                If flagged, coeffs will not be output   
    --no_join                  If flagged, don't join files at end of run
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

from boussinesq_dynamics.equations import BoussinesqEquations2D
from tools.checkpointing import Checkpoint
checkpoint_min = 30
    
def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, aspect=4,
                    fixed_flux=False, fixed_T=True, mixed_flux_T = False,
                    stress_free=False, no_slip=True,
                    viscous_heating=False, restart=None,
                    run_time=23.5, run_time_buoyancy=50, run_time_iter=np.inf,
                    max_writes=20, max_slice_writes=20,
                    data_dir='./', coeff_output=True, verbose=False, no_join=False):
    import os
    from dedalus.tools.config import config
    
    config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
    config['logging']['file_level'] = 'DEBUG'
    
    import mpi4py.MPI
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.makedirs('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    logger = logging.getLogger(__name__)
    logger.info("saving run in: {}".format(data_dir))

    import time
    from dedalus import public as de
    from dedalus.extras import flow_tools
    from dedalus.tools  import post
    
    # input parameters
    logger.info("Ra = {}, Pr = {}".format(Rayleigh, Prandtl))

    equations = BoussinesqEquations2D(stream_function=stress_free)

    # Parameters
    Lz = 1.
    Lx = aspect*Lz
    if nx is None:
        nx = int(nz*aspect)
    logger.info("resolution: [{}x{}]".format(nx, nz))

    equations.set_domain(nx=nx, nz=nz, Lx=Lx, Lz=Lz)
    equations.set_IVP(Rayleigh, Prandtl)

    bc_dict = { 'fixed_flux'              :   None,
                'fixed_temperature'       :   None,
                'mixed_flux_temperature'  :   None,
                'mixed_temperature_flux'  :   None,
                'stress_free'             :   None,
                'no_slip'                 :   None }
    if fixed_flux:
        bc_dict['fixed_flux'] = True
    elif fixed_T:
        bc_dict['fixed_temperature'] = True
    elif mixed_flux_T:
        bc_dict['mixed_flux_temperature'] = True

    if stress_free:
        bc_dict['stress_free'] = True
    elif no_slip:
        bc_dict['no_slip'] = True

    equations.set_BC(**bc_dict)

    # Build solver
    ts = de.timesteppers.RK443
    cfl_safety = 1

    solver = equations.problem.build_solver(ts)
    logger.info('Solver built')

    checkpoint = Checkpoint(data_dir)
    if restart is None:
        equations.set_IC(solver)
        dt = None
        mode = 'overwrite'
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        mode = 'append'
    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
        
    # Integration parameters
    solver.stop_sim_time  = run_time_buoyancy
    solver.stop_wall_time = run_time*3600.
    solver.stop_iteration = run_time_iter

    # Analysis
    equations.initialize_output(solver, data_dir, coeff_output=coeff_output)

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=0.1, threshold=0.1)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')

    first_step = True
    # Main loop
    try:
        logger.info('Starting loop')
        Re_avg = 0
        while solver.ok and np.isfinite(Re_avg):
            dt = CFL.compute_dt()
            solver.step(dt) #, trim=True)
            Re_avg = flow.grid_average('Re')
            log_string =  'Iteration: {:5d}, '.format(solver.iteration)
            log_string += 'Time: {:8.3e}, dt: {:8.3e}, '.format(solver.sim_time, dt)
            log_string += 'Re: {:8.3e}/{:8.3e}'.format(Re_avg, flow.max('Re'))
            logger.info(log_string)
            
            if first_step:
                if verbose:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    ax = fig.add_subplot(1,1,1)
                    ax.spy(solver.pencils[0].L, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern.png", dpi=1200)
                    
                    import scipy.sparse.linalg as sla
                    LU = sla.splu(solver.pencils[0].LHS.tocsc(), permc_spec='NATURAL')
                    fig = plt.figure()
                    ax = fig.add_subplot(1,2,1)
                    ax.spy(LU.L.A, markersize=1, markeredgewidth=0.0)
                    ax = fig.add_subplot(1,2,2)
                    ax.spy(LU.U.A, markersize=1, markeredgewidth=0.0)
                    fig.savefig(data_dir+"sparsity_pattern_LU.png", dpi=1200)
                    
                    logger.info("{} nonzero entries in LU".format(LU.nnz))
                    logger.info("{} nonzero entries in LHS".format(solver.pencils[0].LHS.tocsc().nnz))
                    logger.info("{} fill in factor".format(LU.nnz/solver.pencils[0].LHS.tocsc().nnz))
                first_step=False
                start_time = time.time()
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        main_loop_time = end_time-start_time
        n_iter_loop = solver.iteration-1
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*equations.domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))

        final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', iter=1)
        final_checkpoint.add_system(solver.state, layout='c')
        solver.step(dt) #clean this up in the future...works for now.
        post.merge_analysis(data_dir+'final_checkpoint')
        
        if not no_join:
            logger.info('beginning join operation')
            post.merge_analysis(data_dir+'checkpoints')

            for task in analysis_tasks:
                logger.info(task.base_path)
                post.merge_analysis(task.base_path)

        logger.info(40*"=")
        logger.info('Iterations: {:d}'.format(n_iter_loop))
        logger.info('Sim end time: {:f}'.format(solver.sim_time))
        logger.info('Run time: {:f} sec'.format(main_loop_time))
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*equations.domain.dist.comm_cart.size))
        logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    import logging
    logger = logging.getLogger(__name__)

    from numpy import inf as np_inf
    
    import sys

    fixed_flux = args['--fixed_flux']
    fixed_T = args['--fixed_T']
    mixed_flux_T = args['--mixed_flux_T']
    if not fixed_flux or mixed_flux_T:
        fixed_T = True


    stress_free = args['--stress_free']
    no_slip = args['--no_slip']
    if not stress_free:
        no_slip = True

    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    if fixed_flux:
        data_dir += '_flux'
    elif mixed_flux_T:
        data_dir += '_mixed'


    if args['--viscous_heating']:
        data_dir += '_visc'
    data_dir += "_Ra{}_Pr{}_a{}".format(args['--Rayleigh'], args['--Prandtl'], args['--aspect'])
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))

    if args['--nx'] is not None:
        nx = int(args['--nx'])
    else:
        nx = None

    if args['--run_time_iter'] is not None:
        run_time_iter = int(float(args['--run_time_iter']))
    else:
        run_time_iter = np_inf        

    run_time_buoy = args['--run_time_buoy']
    if not isinstance(run_time_buoy, type(None)):
        run_time_buoy = float(run_time_buoy)
    else:
        run_time_buoy = np.inf
        
    Rayleigh_Benard(Rayleigh=float(args['--Rayleigh']),
                    Prandtl=float(args['--Prandtl']),
                    restart=(args['--restart']),
                    aspect=int(args['--aspect']),
                    nz=int(args['--nz']),
                    nx=nx,
                    fixed_flux=fixed_flux, fixed_T=fixed_T,
                    mixed_flux_T=mixed_flux_T,
                    no_slip=no_slip, stress_free=stress_free,
                    viscous_heating=args['--viscous_heating'],
                    run_time=float(args['--run_time']),
                    run_time_buoyancy=run_time_buoy,
                    run_time_iter=run_time_iter,
                    data_dir=data_dir,
                    max_writes=int(args['--max_writes']),
                    max_slice_writes=int(args['--max_slice_writes']),
                    coeff_output=not(args['--no_coeffs']),
                    verbose=args['--verbose'],
                    no_join=args['--no_join'])
    


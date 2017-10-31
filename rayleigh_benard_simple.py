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
    --run_time_therm=<run_time_therm> Run time, in thermal times [default: 1]
    --run_time_iter=<run_time_iter>   Run time, number of iterations; if not set, n_iter=np.inf

    --restart=<restart_file>   Restart from checkpoint

    --max_writes=<max_writes>              Writes per file for files other than slices and coeffs [default: 20]
    --max_slice_writes=<max_slice_writes>  Writes per file for slices and coeffs [default: 20]
    
    --label=<label>            Optional additional case name label
    --verbose                  Do verbose output (e.g., sparsity patterns of arrays)
    --no_coeffs                If flagged, coeffs will not be output   
    --no_join                  If flagged, don't join files at end of run
    --root_dir=<dir>           Root directory for output [default: ./]

    --do_bvp                             If flagged, do BVPs at regular intervals when Re > 1 to converge faster
    --num_bvps=<num>                     Maximum number of BVPs to do [default: 20]
    --bvp_time=<time>                    How often to do a bvp, in tbuoy [default: 5]
    --bvp_equil_time=<time>              How long to wait after a previous BVP before starting to average for next one, in tbuoy [default: 10]
    --bvp_transient_time=<time>          How long to wait at beginning of run before starting to average for next one, in tbuoy [default: 10]
    --bvp_resolution_factor=<mult>       an int, how many times larger than nz should the bvp nz be? [default: 1]

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
from bvps.bvp_tools import BoussinesqBVPSolver
from tools.checkpointing import Checkpoint
checkpoint_min = 30
    
def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, aspect=4,
                    fixed_flux=False, fixed_T=True, mixed_flux_T = False,
                    stress_free=False, no_slip=True,
                    viscous_heating=False, restart=None,
                    run_time=23.5, run_time_buoyancy=None, run_time_iter=np.inf, run_time_therm=1,
                    max_writes=20, max_slice_writes=20,
                    data_dir='./', coeff_output=True, verbose=False, no_join=False,
                    do_bvp=False, bvp_time=20, num_bvps=1, bvp_equil_time=10, bvp_resolution_factor=1,
                    bvp_transient_time=30):
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


    # Parameters
    Lz = 1.
    Lx = aspect*Lz
    if nx is None:
        nx = int(nz*aspect)
    logger.info("resolution: [{}x{}]".format(nx, nz))

    equations = BoussinesqEquations2D(stream_function=stress_free, nx=nx, nz=nz, Lx=Lx, Lz=Lz)
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
    if isinstance(restart, type(None)):
        equations.set_IC(solver)
        dt = None
        mode = 'overwrite'
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        mode = 'append'
    checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, mode=mode)
        
    # Integration parameters
    if not isinstance(run_time_therm, type(None)):
        solver.stop_sim_time = run_time_therm*equations.thermal_time
    elif not isinstance(run_time_buoyancy, type(None)):
        solver.stop_sim_time  = run_time_buoyancy
    else:
        solver.stop_sim_time  = np.inf
    solver.stop_wall_time = run_time*3600.
    solver.stop_iteration = run_time_iter

    # Analysis
    output_dt = 0.25
    max_dt    = output_dt
    analysis_tasks = equations.initialize_output(solver, data_dir, coeff_output=coeff_output, output_dt=output_dt, mode=mode)

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')


    if do_bvp:
        bvp_solver = BoussinesqBVPSolver(BoussinesqEquations2D, nz, \
                                   flow, equations.domain.dist.comm_cart, \
                                   solver, bvp_time, \
                                   num_bvps, bvp_equil_time,
                                   bvp_transient_time=bvp_transient_time)
        bc_dict.pop('stress_free')
        bc_dict.pop('no_slip')

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
            log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time, solver.sim_time/equations.thermal_time,  dt)
            log_string += 'Re: {:8.3e}/{:8.3e}'.format(Re_avg, flow.max('Re'))
            logger.info(log_string)

            if do_bvp:
                bvp_solver.update_avgs(dt, min_Re=1e0)
                if bvp_solver.check_if_solve():
                    atmo_kwargs = { 'stream_function' : stress_free,
                                    'nz'              : nz*bvp_resolution_factor,
                                    'Lz'              : Lz
                                   }
                    diff_args = [Rayleigh, Prandtl]
                    bvp_solver.solve_BVP(atmo_kwargs, diff_args, bc_dict)


            
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
        try:
            final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
            final_checkpoint.set_checkpoint(solver, wall_dt=1, mode="append")
            solver.step(dt) #clean this up in the future...works for now.
            post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
        except:
            print('cannot save final checkpoint')
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
    if not (fixed_flux or mixed_flux_T):
        fixed_T = True


    stress_free = args['--stress_free']
    no_slip = args['--no_slip']
    if not stress_free:
        no_slip = True

    # save data in directory named after script
    data_dir = args['--root_dir'] + sys.argv[0].split('.py')[0]
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
    run_time_therm = args['--run_time_therm']
    if not isinstance(run_time_therm, type(None)):
        run_time_therm = float(run_time_therm)
        
    Rayleigh_Benard(Rayleigh=float(args['--Rayleigh']),
                    Prandtl=float(args['--Prandtl']),
                    restart=args['--restart'],
                    aspect=int(args['--aspect']),
                    nz=int(args['--nz']),
                    nx=nx,
                    fixed_flux=fixed_flux, fixed_T=fixed_T,
                    mixed_flux_T=mixed_flux_T,
                    no_slip=no_slip, stress_free=stress_free,
                    viscous_heating=args['--viscous_heating'],
                    run_time=float(args['--run_time']),
                    run_time_buoyancy=run_time_buoy,
                    run_time_therm=run_time_therm,
                    run_time_iter=run_time_iter,
                    data_dir=data_dir,
                    max_writes=int(args['--max_writes']),
                    max_slice_writes=int(args['--max_slice_writes']),
                    coeff_output=not(args['--no_coeffs']),
                    verbose=args['--verbose'],
                    no_join=args['--no_join'],
                    do_bvp=args['--do_bvp'],
                     bvp_time=float(args['--bvp_time']),
                     num_bvps=int(args['--num_bvps']),
                     bvp_equil_time=float(args['--bvp_equil_time']),
                     bvp_transient_time=float(args['--bvp_transient_time']),
                     bvp_resolution_factor=int(args['--bvp_resolution_factor']))
    


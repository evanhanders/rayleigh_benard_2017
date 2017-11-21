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
    --fixed_T                  Fixed temperature boundary conditions top/bottom; default if no choice is made
    
    --run_time=<run_time>             Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_bouy>   Run time, in buoyancy times [default: 50]
    --run_time_iter=<run_time_iter>   Run time, number of iterations; if not set, n_iter=np.inf

    --restart=<restart_file>   Restart from checkpoint

    --max_writes=<max_writes>              Writes per file for files other than slices and coeffs [default: 10]
    --max_slice_writes=<max_slice_writes>  Writes per file for slices and coeffs [default: 10]
    
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
try:
    from dedalus.extras.checkpointing import Checkpoint
    checkpointing = True
except:
    logger.info("No checkpointing available; disabling capability")
    checkpointing = False
    
def global_noise(domain, seed=42, scale=None, **kwargs):            
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    filter_field(noise_field, **kwargs)
    if scale is not None:
        noise_field.set_scales(scale, keep_data=True)
        
    return noise_field['g']

def filter_field(field,frac=0.5):
    logger.info("filtering field with frac={}".format(frac))
    dom = field.domain
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    coeff = []
    for i in range(dom.dim)[::-1]:
        coeff.append(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False))
    cc = np.meshgrid(*coeff)

    field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

    for i in range(dom.dim):
        field_filter = field_filter | (cc[i][local_slice] > frac)
    field['c'][field_filter] = 0j

def Rayleigh_Benard(Rayleigh=1e6, Prandtl=1, nz=64, nx=None, aspect=4,
                    fixed_flux=False, fixed_T=True,
                    viscous_heating=False, restart=None,
                    run_time=23.5, run_time_buoyancy=50, run_time_iter=np.inf,
                    max_writes=10, max_slice_writes=10,
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
            
    # Parameters
    Lz = 1.
    Lx = aspect*Lz

    if nx is None:
        nx = int(nz*aspect)

    logger.info("resolution: [{}x{}]".format(nx, nz))
    # Create bases and domain
    x_basis = de.Fourier('x',   nx, interval=(0, Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    if fixed_flux:
        T_bc_var = 'Tz'
    elif fixed_T:
        T_bc_var = 'T'
                
    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['Tz','T','p','u','w','Oy'])
    problem.meta['p',T_bc_var,'Oy','w']['z']['dirichlet'] = True

    T0_z = domain.new_field()
    T0_z.meta['x']['constant'] = True
    T0_z['g'] = -1
    T0 = domain.new_field()
    T0.meta['x']['constant'] = True
    T0['g'] = Lz/2-domain.grid(-1)
    problem.parameters['T0'] = T0
    problem.parameters['T0_z'] = T0_z
    
    problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
    problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
    problem.parameters['F'] = F = 1
    
    problem.parameters['Lx'] = Lx
    problem.parameters['Lz'] = Lz
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
    
    problem.substitutions['v'] = '0'
    problem.substitutions['Ox'] = '0'
    problem.substitutions['Oz'] = '(dx(v) )'
    problem.substitutions['Kx'] = '( -dz(Oy))'
    problem.substitutions['Ky'] = '(dz(Ox) - dx(Oz))'
    problem.substitutions['Kz'] = '(dx(Oy) )'
    
    problem.substitutions['vorticity'] = 'Oy' 
    problem.substitutions['enstrophy'] = 'Oy**2'

    problem.substitutions['u_fluc'] = '(u - plane_avg(u))'
    problem.substitutions['w_fluc'] = '(w - plane_avg(w))'
    problem.substitutions['KE'] = '(0.5*(u*u+w*w))'
    
    problem.substitutions['sigma_xz'] = '(dx(w) + Oy + dx(w))'
    problem.substitutions['sigma_xx'] = '(2*dx(u))'
    problem.substitutions['sigma_zz'] = '(2*dz(w))'

    if viscous_heating:
        problem.substitutions['visc_heat']   = 'R*(sigma_xz**2 + sigma_xx*dx(u) + sigma_zz*dz(w))'
        problem.substitutions['visc_flux_z'] = 'R*(u*sigma_xz + w*sigma_zz)'
    else:
        problem.substitutions['visc_heat']   = '0'
        problem.substitutions['visc_flux_z'] = '0'
        
    problem.substitutions['conv_flux_z'] = '(w*T + visc_flux_z)/P'
    problem.substitutions['kappa_flux_z'] = '(-Tz)'

    
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("dt(T) - P*(dx(dx(T)) + dz(Tz)) + w*T0_z    = -(u*dx(T) + w*Tz)  - visc_heat")
    # O == omega = curl(u);  K = curl(O)
    problem.add_equation("dt(u)  + R*Kx  + dx(p)              =  v*Oz - w*Oy ")
    problem.add_equation("dt(w)  + R*Kz  + dz(p)    -T        =  u*Oy - v*Ox ")
    problem.add_equation("dx(u) + dz(w) = 0")
    problem.add_equation("Oy - dz(u) + dx(w) = 0")

    problem.add_bc("right(p) = 0", condition="(nx == 0)")
    if fixed_flux:
        problem.add_bc("left(Tz)  = 0")
        problem.add_bc("right(Tz) = 0")
    elif fixed_T:
        problem.add_bc("left(T)  = 0")
        problem.add_bc("right(T) = 0")
    problem.add_bc("left(Oy) = 0")
    problem.add_bc("right(Oy) = 0")
    problem.add_bc("left(w)  = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")

    # Build solver
    ts = de.timesteppers.RK443
    cfl_safety = 1
    
    solver = problem.build_solver(ts)
    logger.info('Solver built')
        
    checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoints', wall_dt=8*3600, max_writes=1)
    checkpoint.add_system(solver.state, layout='c')

    
    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    T = solver.state['T']
    Tz = solver.state['Tz']

    # Random perturbations, initialized globally for same results in parallel
    noise = global_noise(domain, scale=1, frac=0.25)

    if restart is None:
        # Linear background + perturbations damped at walls
        zb, zt = z_basis.interval
        pert =  1e-3 * noise * (zt - z) * (z - zb)
        T['g'] = pert
        T.differentiate('z', out=Tz)
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)
        
    # Integration parameters
    solver.stop_sim_time  = run_time_buoyancy
    solver.stop_wall_time = run_time*3600.
    solver.stop_iteration = run_time_iter

    # Analysis
    analysis_tasks = []
    snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=0.1, max_writes=max_slice_writes)
    snapshots.add_task("T + T0", name='T')
    snapshots.add_task("enstrophy")
    snapshots.add_task("vorticity")
    analysis_tasks.append(snapshots)

    snapshots_small = solver.evaluator.add_file_handler(data_dir+'slices_small', sim_dt=0.1, max_writes=max_slice_writes)
    snapshots_small.add_task("T + T0", name='T', scales=0.25)
    snapshots_small.add_task("enstrophy", scales=0.25)
    snapshots_small.add_task("vorticity", scales=0.25)
    analysis_tasks.append(snapshots_small)

    if coeff_output:
        coeffs = solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=0.1, max_writes=max_slice_writes)
        coeffs.add_task("T+T0", name="T", layout='c')
        coeffs.add_task("T - plane_avg(T)", name="T'", layout='c')
        coeffs.add_task("w", layout='c')
        coeffs.add_task("u", layout='c')
        coeffs.add_task("enstrophy", layout='c')
        coeffs.add_task("vorticity", layout='c')
        analysis_tasks.append(coeffs)

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=0.1, max_writes=max_writes)
    profiles.add_task("plane_avg(T+T0)", name="T")
    profiles.add_task("plane_avg(T)", name="T'")
    profiles.add_task("plane_avg(u)", name="u")
    profiles.add_task("plane_avg(w)", name="w")
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    # This may have an error:
    profiles.add_task("plane_avg(conv_flux_z)/plane_avg(kappa_flux_z) + 1", name="Nu")
    profiles.add_task("plane_avg(conv_flux_z) + plane_avg(kappa_flux_z)",   name="Nu_2")

    analysis_tasks.append(profiles)

    scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=0.1, max_writes=max_writes)
    scalar.add_task("vol_avg(T)", name="IE")
    scalar.add_task("vol_avg(KE)", name="KE")
    scalar.add_task("vol_avg(-T*z)", name="PE_fluc")
    scalar.add_task("vol_avg(T) + vol_avg(KE) + vol_avg(-T*z)", name="TE")
    scalar.add_task("0.5*vol_avg(u_fluc*u_fluc+w_fluc*w_fluc)", name="KE_fluc")
    scalar.add_task("0.5*vol_avg(u*u)", name="KE_x")
    scalar.add_task("0.5*vol_avg(w*w)", name="KE_z")
    scalar.add_task("0.5*vol_avg(u_fluc*u_fluc)", name="KE_x_fluc")
    scalar.add_task("0.5*vol_avg(w_fluc*w_fluc)", name="KE_z_fluc")
    scalar.add_task("vol_avg(plane_avg(u)**2)", name="u_avg")
    scalar.add_task("vol_avg((u - plane_avg(u))**2)", name="u1")
    scalar.add_task("vol_avg(conv_flux_z) + 1.", name="Nu")
    analysis_tasks.append(scalar)

    # workaround for issue #29
    problem.namespace['enstrophy'].store_last = True

    # CFL
    CFL = flow_tools.CFL(solver, initial_dt=0.1, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.5, max_dt=0.1, threshold=0.1)
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("sqrt(u*u + w*w) / R", name='Re')

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
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
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
        logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
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
    if not fixed_flux:
        fixed_T = True

    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    if fixed_flux:
        data_dir += '_flux'
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
        
    Rayleigh_Benard(Rayleigh=float(args['--Rayleigh']),
                    Prandtl=float(args['--Prandtl']),
                    restart=(args['--restart']),
                    aspect=int(args['--aspect']),
                    nz=int(args['--nz']),
                    nx=nx,
                    fixed_flux=fixed_flux, fixed_T=fixed_T,
                    viscous_heating=args['--viscous_heating'],
                    run_time=float(args['--run_time']),
                    run_time_buoyancy=float(args['--run_time_buoy']),
                    run_time_iter=run_time_iter,
                    data_dir=data_dir,
                    max_writes=int(args['--max_writes']),
                    max_slice_writes=int(args['--max_slice_writes']),
                    coeff_output=not(args['--no_coeffs']),
                    verbose=args['--verbose'],
                    no_join=args['--no_join'])
    


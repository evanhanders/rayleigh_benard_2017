"""
    This file is a partial driving script for boussinesq dynamics.  Here,
    formulations of the boussinesq equations are handled in a clean way using
    classes.
"""
import numpy as np
from mpi4py import MPI
import scipy.special as scp

from collections import OrderedDict

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Equations():
    """
    A general, abstract class for solving equations in dedalus.

    This class can be inherited by other classes to set up specific equation sets, but
    the base (parent) class contains much of the logic we will need regardless (setting
    up the domain, creating a problem or a new non-constant coefficient, etc.)

    Attributes:
        compound        - If True, z-basis is a set of compound chebyshevs
        dimensions      - The dimensionality of the problem (1D, 2D, 3D)
        domain          - The dedalus domain on which the problem is being solved
        mesh            - The processor mesh over which the problem is being solved
        problem         - The Dedalus problem object that is being solved
        problem_type    - The type of problem being solved (IVP, EVP)
        x, y, z         - 1D NumPy arrays containing the physical coordinates of grid points in grid space
        Lx, Ly, Lz      - Scalar containing the size of the atmosphere in x, y, z directions
        nx, ny, nz      - Scalars containing the number of points in the x, y, z directions
        delta_x, delta_y- Grid spacings in the x, y directions (assuming constant grid spacing)
        z_dealias       - 1D NumPy array containing the dealiased locations of grid points in the z-direction
    """
    def __init__(self, dimensions=2):
        """Initialize all object attributes"""
        self.compound       = False
        self.dimensions     = dimensions
        self.domain         = None
        self.mesh           = None
        self.problem        = None
        self.problem_type   = ''
        self.x, self.Lx, self.nx, self.delta_x   = [None]*4
        self.y, self.Ly, self.ny, self.delta_y   = [None]*4
        self.z, self.z_dealias, self.Lz, self.nz = [None]*4
        return

    def _set_domain(self, nx=256, Lx=4,
                          ny=256, Ly=4,
                          nz=128, Lz=1,
                          grid_dtype=np.float64, comm=MPI.COMM_WORLD, mesh=None):
        """
        Here the dedalus domain is created for the equation set

        Inputs:
            nx, ny, nz      - Number of grid points in the x, y, z directions
            Lx, Ly, Lz      - Physical size of the x, y, z direction
            grid_dtype      - Datatype to use for grid points in the problem
            comm            - Comm group over which to solve.  Use COMM_SELF for EVP
            mesh            - The processor mesh over which the problem is solved.
        """
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        self.mesh=mesh
        
        if not isinstance(nz, list):
            nz = [nz]
        if not isinstance(Lz, list):
            Lz = [Lz]   

        if len(nz)>1:
            logger.info("Setting compound basis in vertical (z) direction")
            z_basis_list = []
            Lz_interface = 0.
            for iz, nz_i in enumerate(nz):
                Lz_top = Lz[iz]+Lz_interface
                z_basis = de.Chebyshev('z', nz_i, interval=[Lz_interface, Lz_top], dealias=3/2)
                z_basis_list.append(z_basis)
                Lz_interface = Lz_top
            self.compound = True
            z_basis = de.Compound('z', tuple(z_basis_list),  dealias=3/2)
        elif len(nz)==1:
            logger.info("Setting single chebyshev basis in vertical (z) direction")
            z_basis = de.Chebyshev('z', nz[0], interval=[0, Lz[0]], dealias=3/2)
        
        if self.dimensions > 1:
            x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        if self.dimensions > 2:
            y_basis = de.Fourier(  'y', ny, interval=[0., Ly], dealias=3/2)
        if self.dimensions == 1:
            bases = [z_basis]
        elif self.dimensions == 2:
            bases = [x_basis, z_basis]
        elif self.dimensions == 3:
            bases = [x_basis, y_basis, z_basis]
        else:
            logger.error('>3 dimensions not implemented')
        
        self.domain = de.Domain(bases, grid_dtype=grid_dtype, comm=comm, mesh=mesh)
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)

        if self.dimensions == 1:
            self.Lx, self.Ly = 0, 0
        if self.dimensions > 1:
            self.x = self.domain.grid(0)
            self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.nx = self.domain.bases[0].coeff_size
            self.delta_x = self.Lx/self.nx
        if self.dimensions > 2:
            self.y = self.domain.grid(1)
            self.Ly = self.domain.bases[1].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.ny = self.domain.bases[1].coeff_size
            self.delta_y = self.Ly/self.ny
    
    def set_IVP(self, *args, ncc_cutoff=1e-10, **kwargs):
        """
        Constructs and initial value problem of the current object's equation set
        """
        self.problem_type = 'IVP'
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_EVP(self, *args, ncc_cutoff=1e-10, tolerance=1e-10, **kwargs):
        """
        Constructs an eigenvalue problem of the current objeect's equation set.
        Note that dt(f) = omega * f, not i * omega * f, so real parts of omega
        are growth / shrinking nodes, imaginary parts are oscillating.
        """

        self.problem_type = 'EVP'
        self.problem = de.EVP(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff, tolerance=tolerance)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def set_equations(self, *args, **kwargs):
        """ This function must be implemented in child objects of this class """
        pass

    def get_problem(self):
        return self.problem

    def _new_ncc(self):
        """
        Create a new field of the atmosphere from the dedalus domain. Field's metadata is
        set so that it is constant in the x- and y- directions (but can vary in the z).
        """
        # is this used at all in equations.py (other than rxn), or just in atmospheres?
        # the naming conventions here force cartesian, generalize to spheres etc. make sense?
        # should "necessary quantities" logic occur here?
        field = self.domain.new_field()
        if self.dimensions > 1:
            field.meta['x']['constant'] = True
        if self.dimensions > 2:
            field.meta['y']['constant'] = True            
        return field

    def _new_field(self):
        """Create a new field of the atmosphere that is NOT a NCC. """
        field = self.domain.new_field()
        return field

    def _set_subs(self):
        """ This function must be implemented in child objects of this class """
        pass

    def global_noise(self, seed=42, **kwargs):
        """
        Create a field fielled with random noise of order 1.  Modify seed to
        get varying noise, keep seed the same to directly compare runs.
        """
        # Random perturbations, initialized globally for same results in parallel
        gshape = self.domain.dist.grid_layout.global_shape(scales=self.domain.dealias)
        slices = self.domain.dist.grid_layout.slices(scales=self.domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field = self._new_field()
        noise_field.set_scales(self.domain.dealias, keep_data=False)
        noise_field['g'] = noise
        self.filter_field(noise_field, **kwargs)

        return noise_field

    def filter_field(self, field, frac=0.25):
        """
        Filter a field in coefficient space by cutting off all coefficient above
        a given threshold.  This is accomplished by changing the scale of a field,
        forcing it into coefficient space at that small scale, then coming back to
        the original scale.

        Inputs:
            field   - The dedalus field to filter
            frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                        The upper 75% of coefficients are set to 0.
        """
        dom = field.domain
        logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
        orig_scale = field.meta[:]['scale']
        field.set_scales(frac, keep_data=True)
        field['c']
        field['g']
        field.set_scales(orig_scale, keep_data=True)

    
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
    scalar.add_task("vol_avg(T) + vol_avg(KE)", name="TE")
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


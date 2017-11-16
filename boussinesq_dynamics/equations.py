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

    def set_domain(self, nx=256, Lx=4,
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


class BoussinesqEquations2D(Equations):
    """
    An extension of the Equations class which contains the full 2D form of the boussinesq
    equations.   
    """
    def __init__(self,*args,  dimensions=2, **kwargs):
#        self.stream_function = stream_function
        super(BoussinesqEquations2D, self).__init__(dimensions=dimensions)
#        self.variables=['T1_z','T1','p','u','w','Oy']
#        else:
        self.variables=['p','T1','u','w','T1_z','uz','wz']

        self.set_domain(*args, **kwargs)

    def _set_parameters(self, Rayleigh, Prandtl):
        """
        Set up important parameters of the problem for boussinesq convection
        """
        self.T0_z      = self._new_ncc()
        self.T0_z['g'] = -1
        self.T0        = self._new_ncc()
        self.T0['g']   = self.Lz/2 - self.domain.grid(-1)
        self.T0_zz = self._new_ncc()
        self.T0_zz['g'] = 0
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['T0_z'] = self.T0_z
        self.problem.parameters['T0_zz'] = self.T0_zz

        # Characteristic scales (things we've non-dimensionalized on
        self.problem.parameters['t_buoy']   = 1.
        self.problem.parameters['v_ff']     = 1.

        self.problem.parameters['Rayleigh'] = Rayleigh
        self.problem.parameters['Prandtl']  = Prandtl

        self.P = (Rayleigh * Prandtl)**(-1./2)
        self.R = (Rayleigh / Prandtl)**(-1./2)
        self.problem.parameters['P'] = (Rayleigh * Prandtl)**(-1./2)
        self.problem.parameters['R'] = (Rayleigh / Prandtl)**(-1./2)
        self.thermal_time = (Rayleigh / Prandtl)**(1./2)

        self.problem.parameters['Lx'] = self.Lx
        self.problem.parameters['Lz'] = self.Lz
       
    def _set_subs(self, viscous_heating=False):

        if self.dimensions == 1:
            self.problem.substitutions['plane_avg(A)'] = 'A'
            self.problem.substitutions['plane_std(A)'] = '0'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lz'
        elif self.dimensions == 2:
            self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
            self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
            self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
        self.problem.substitutions['UdotGrad(A, A_z)'] = '(u * dx(A) + w * A_z)'
        self.problem.substitutions['Lap(A, A_z)'] = '(dx(dx(A)) + dz(A_z))'
       
        self.problem.substitutions['v'] = '0'
        self.problem.substitutions['Ox'] = '0'
        self.problem.substitutions['Oy'] = "(uz - dx(w))"
        self.problem.substitutions['Oz'] = '(dx(v) )'
        self.problem.substitutions['Kx'] = '( -dz(Oy))'
        self.problem.substitutions['Ky'] = '(dz(Ox) - dx(Oz))'
        self.problem.substitutions['Kz'] = '(dx(Oy) )'

        #Diffusivities; diffusive timescale
        self.problem.substitutions['chi']= '(v_ff * Lz * P)'
        self.problem.substitutions['nu'] = '(v_ff * Lz * R)'
        self.problem.substitutions['t_therm'] = '(Lz**2/chi)'
        
        self.problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['vorticity'] = 'Oy' 
        self.problem.substitutions['enstrophy'] = 'Oy**2'

        self.problem.substitutions['u_fluc'] = '(u - plane_avg(u))'
        self.problem.substitutions['w_fluc'] = '(w - plane_avg(w))'
        self.problem.substitutions['KE'] = '(0.5*vel_rms**2)'

        self.problem.substitutions['Re'] = '(vel_rms / nu)'
        self.problem.substitutions['Pe'] = '(vel_rms / chi)'
        
        self.problem.substitutions['sigma_xz'] = '(dx(w) + Oy + dx(w))'
        self.problem.substitutions['sigma_xx'] = '(2*dx(u))'
        self.problem.substitutions['sigma_zz'] = '(2*dz(w))'

        if viscous_heating:
            self.problem.substitutions['visc_heat_R'] = '(-1 * R * vorticity**2 - w * T1)'
#            self.problem.substitutions['visc_heat']   = 'R*(sigma_xz**2 + sigma_xx*dx(u) + sigma_zz*dz(w))'
            self.problem.substitutions['visc_flux_z'] = 'R*(u*sigma_xz + w*sigma_zz)'
        else:
            self.problem.substitutions['visc_heat_R']   = '0'
            self.problem.substitutions['visc_flux_z'] = '0'
            
        self.problem.substitutions['conv_flux_z'] = '(w*T1 + visc_flux_z)/P'
        self.problem.substitutions['kappa_flux_z'] = '(-T1_z)'


       
    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None):

        self.dirichlet_set = []

        self.set_thermal_BC(fixed_flux=fixed_flux, fixed_temperature=fixed_temperature,
                            mixed_flux_temperature=mixed_flux_temperature, mixed_temperature_flux=mixed_temperature_flux)
        
        self.set_velocity_BC(stress_free=stress_free, no_slip=no_slip)
        
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True
            
    def set_thermal_BC(self, fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None):
        # not(None) logic is going to be deprecated in future python releases.  What is the best way to use None as a function argument and in logic?  "if A is None" vs "if not(A)" and "if A".  Gabo will check.
        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True

        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (full form)")
            self.problem.add_bc( "left(T1_z) = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: fixed flux/fixed temperature")
            self.problem.add_bc("left(T1_z) = 0")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: fixed temperature/fixed flux")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise

    def set_velocity_BC(self, stress_free=None, no_slip=None):
        if not(stress_free) and not(no_slip):
            stress_free = True
            
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc("left(Oy) = 0")
            self.problem.add_bc("right(Oy) = 0")
            self.dirichlet_set.append('Oy')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        if self.dimensions > 1:
            self.problem.add_bc("right(p) = 0", condition="(nx == 0)")
            self.problem.add_bc("right(w) = 0", condition="(nx != 0)")
        else:
            self.problem.add_bc("right(p) = 0")
        self.dirichlet_set.append('w')
        
    def set_IC(self, solver, A0=1e-6, **kwargs):
        # initial conditions
        T_IC = solver.state['T1']
        T_z_IC = solver.state['T1_z']
            
        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        T_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T_IC['g'] = A0*np.sin(np.pi*self.z_dealias/self.Lz)*noise['g']*self.T0['g']
        T_IC.differentiate('z', out=T_z_IC)
        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))


    def set_equations(self, Rayleigh, Prandtl, kx = 0, viscous_heating = False):
        # 2D Boussinesq hydrodynamics
        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
 
        self._set_parameters(Rayleigh, Prandtl)
        self._set_subs(viscous_heating=viscous_heating)


#        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z   = -UdotGrad(T1, T1_z)  + visc_heat_R")
#        self.problem.add_equation("dt(u)  + R*Kx  + dx(p)              =  v*Oz - w*Oy ")
#        self.problem.add_equation("dt(w)  + R*Kz  + dz(p)    - T1      =  u*Oy - v*Ox ")
#        self.problem.add_equation("dx(u) + dz(w) = 0")
#        self.problem.add_equation("Oy - dz(u) + dx(w) = 0")
#        self.problem.add_equation("T1_z - dz(T1) = 0")
#        else:
        self.problem.add_equation("dx(u) + wz = 0")
        self.problem.add_equation("dt(T1) - P*Lap(T1, T1_z) + w*T0_z  = -UdotGrad(T1, T1_z) + visc_heat_R")
        self.problem.add_equation("dt(u)  - R*Lap(u, uz) + dx(p)      = -UdotGrad(u, uz)")
        self.problem.add_equation("dt(w)  - R*Lap(w, wz) + dz(p) - T1 = -UdotGrad(w, wz)")
        self.problem.add_equation("T1_z - dz(T1) = 0")
        self.problem.add_equation("uz - dz(u) = 0")
        self.problem.add_equation("wz - dz(w) = 0")

    def initialize_output(self, solver, data_dir, coeff_output=False,
                          max_writes=20, max_slice_writes=20, output_dt=0.25,
                          mode="overwrite", **kwargs):

        # Analysis
        analysis_tasks = []
        snapshots = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
        snapshots.add_task("T1 + T0", name='T')
        snapshots.add_task("enstrophy")
        snapshots.add_task("vel_rms")
        snapshots.add_task("u")
        snapshots.add_task("w")
        analysis_tasks.append(snapshots)

        if coeff_output:
            coeffs = solver.evaluator.add_file_handler(data_dir+'coeffs', sim_dt=output_dt, max_writes=max_slice_writes, mode=mode)
            coeffs.add_task("T1+T0", name="T", layout='c')
            coeffs.add_task("T1 - plane_avg(T1)", name="T'", layout='c')
            coeffs.add_task("w", layout='c')
            coeffs.add_task("u", layout='c')
            coeffs.add_task("enstrophy", layout='c')
            coeffs.add_task("vorticity", layout='c')
            analysis_tasks.append(coeffs)

        profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        profiles.add_task("plane_avg(T1+T0)", name="T")
        profiles.add_task("plane_avg(T1)", name="T'")
        profiles.add_task("plane_avg(u)", name="u")
        profiles.add_task("plane_avg(w)", name="w")
        profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
        # This may have an error:
        profiles.add_task("plane_avg(conv_flux_z)/plane_avg(kappa_flux_z) + 1", name="Nu")
        profiles.add_task("plane_avg(conv_flux_z) + plane_avg(kappa_flux_z)",   name="Nu_2")

        analysis_tasks.append(profiles)

        scalar = solver.evaluator.add_file_handler(data_dir+'scalar', sim_dt=output_dt, max_writes=max_writes, mode=mode)
        scalar.add_task("vol_avg(T1)", name="IE")
        scalar.add_task("vol_avg(KE)", name="KE")
        scalar.add_task("vol_avg(T1) + vol_avg(KE)", name="TE")
        scalar.add_task("0.5*vol_avg(u_fluc*u_fluc+w_fluc*w_fluc)", name="KE_fluc")
        scalar.add_task("0.5*vol_avg(u*u)", name="KE_x")
        scalar.add_task("0.5*vol_avg(w*w)", name="KE_z")
        scalar.add_task("0.5*vol_avg(u_fluc*u_fluc)", name="KE_x_fluc")
        scalar.add_task("0.5*vol_avg(w_fluc*w_fluc)", name="KE_z_fluc")
        scalar.add_task("vol_avg(plane_avg(u)**2)", name="u_avg")
        scalar.add_task("vol_avg((u - plane_avg(u))**2)", name="u1")
        scalar.add_task("vol_avg(conv_flux_z) + 1.", name="Nu")
        scalar.add_task("(1-vol_avg(conv_flux_z))**(-1.)", name="Nu_flux")
        scalar.add_task("vol_avg(Re)", name="Re")
        scalar.add_task("vol_avg(Pe)", name="Pe")
        analysis_tasks.append(scalar)

        return analysis_tasks



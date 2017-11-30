from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


from mpi4py import MPI
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from dedalus import public as de


def bl_profile(z, bl, center=0, upper=False):
    """ from Shishkina et al 2015 PRL, eqn 31. """
    a = 2 * np.pi / ( 3 * np.sqrt(3) )
    xi = (z - center) / bl
    if upper:
        xi *= -1
    part1 = ( np.sqrt(3) / (4 * np.pi) ) * np.log( (1 + a*xi)**3 / (1 + (a*xi)**3) )
    part2 = ( 3 / (2 * np.pi) ) * np.arctan( ( 4 * np.pi / 9 ) * xi - 1 / np.sqrt(3) )
    return part1 + part2 + 1./4

def bl_Tz_profile(z, bl, center=0, upper=False):
    """ Derivative of eqn 31 from Shishkina et al 2015 PRL. Normalized to 1 at the boundary. """
    a = 2 * np.pi / ( 3 * np.sqrt(3) )
    xi = (z - center) / bl
    if upper:
        xi *= -1
    part1 = (1/2.) * ( 1/(1 + a*xi) - a**2 * xi**2 / (1 + (a*xi)**3) )
    part2 = (2/3.) * 1 / (1. + (4*np.pi/9. * xi - 1/np.sqrt(3.))**2)
    return part1 + part2 



class BVPSolverBase:
    """
    A base class for solving a BVP in the middle of a running IVP.

    This class sets up basic functionality for tracking profiles and solving BVPs.
    This is just an abstract class, and must be inherited with specific equation
    sets to work.

    Objects of this class are paired with a dedalus solver which is timestepping forward
    through an IVP.  This class calculates horizontal and time averages of important flow
    fields from that IVP, then uses those as NCCs in a BVP to get a more evolved thermal state

    CLASS VARIABLES
    ---------------
        FIELDS     - An OrderedDict of strings of which the time- and horizontally- averaged 
                     profiles are tracked (and fed into the BVP)
        VARS       - An OrderedDict of variables which will be updated by the BVP
        VEL_VARS   - An OrderedDict of variables which contain info about the velocity field;
                        may be updated by BVP.

    Object Attributes:
    ------------------
        avg_started         - If True, time averages for FIELDS has begun
        avg_time_elapsed    - Amount of IVP simulation time over which averages have been taken so far
        avg_time_start      - Simulation time at which average began
        bvp_equil_time      - Amount of sim time to wait for velocities to converge before starting averages
                                at the beginning of IVP or after a BVP is solved
        bvp_transient_time  - Amount of time to wait at the beginning of the sim
        bvp_run_threshold   - Degree of convergence required on time averages before doing BVP
        bvp_l2_check_time   - How often to check for convergence, in simulation time
        bvp_l2_last_check_time - Last time we checked if avgs were converged
        current_local_avg   - Current value of the local portion of the time average of profiles
        current_local_l2    - The avg of the abs() of the change in the avg profile compared to the previous timestep.
        comm                - COMM_WORLD for IVP
        completed_bvps      - # of BVPs that have been completed during this run
        do_bvp              - If True, average profiles are converged, do BVP.
        final_equil_time    - How long to allow the solution to equilibrate after the final bvp
        first_l2            - If True, we haven't taken an L2 average for convergence yet.
        flow                - A dedalus flow_tools.GlobalFlowProperty object for the IVP solver which is tracking
                                the Reynolds number, and will track FIELDS variables
        min_bvp_time        - Minimum simulation time to wait between BVPs.
        min_avg_dt          - Minimum simulation time to wait between adjusting average profiles.
        n_per_proc          - Number of z-points per core (for parallelization)
        num_bvps            - Total (max) number of BVPs to complete
        nx                  - x-resolution of the IVP grid
        nz                  - z-resolution of the IVP grid
        partial_prof_dict   - a dictionary containing local contributions to the averages of FIELDS
        plot_dir            - A directory to save plots into during BVPs.
        profiles_dict       - a dictionary containing the time/horizontal average of FIELDS
        profiles_dict_last  - a dictionary containing the time/horizontal average of FIELDS from the previous bvp
        profiles_dict_curr  - a dictionary containing the time/horizontal average of FIELDS for current atmosphere state
        rank                - comm rank
        size                - comm size
        solver              - The corresponding dedalus IVP solver object
        solver_states       - The states of VARS in solver

    """
    
    FIELDS = None
    VARS   = None
    VEL_VARS   = None

    def __init__(self, nx, nz, flow, comm, solver, num_bvps, bvp_equil_time, bvp_transient_time=20,
                 bvp_run_threshold=1e-2, bvp_l2_check_time=1, min_bvp_time=20, plot_dir=None,
                 min_avg_dt=0.05, final_equil_time = None):
        """
        Initializes the object; grabs solver states and makes room for profile averages
        
        Arguments:
        nx                  - the horizontal resolution of the IVP
        nz                  - the vertical resolution of the IVP
        flow                - a dedalus.extras.flow_tools.GlobalFlowProperty for the IVP solver
        comm                - An MPI comm object for the IVP solver
        solver              - The IVP solver
        min_bvp_time        - Minimum sim time to do average over before doing a bvp
        min_avg_dt          - Minimum sim time to wait between adjusting averages
        num_bvps            - Maximum number of BVPs to solve
        bvp_equil_time      - Sim time to wait after a bvp before beginning averages for the next one
        bvp_transient_time  - Sim time to wait at beginning of simulation before starting average
        bvp_run_threshold   - Level of convergence that must be reached in statistical averages
                                before doing a BVP (1e-2 = 1% variation OK, 1e-3 = 0.1%, so on)
        bvp_l2_check_time   - Sim time to wait between communications to see if we're converged
                                (so that we don't "check for convergence" on all processes at every timestep)
        plot_dir            - If not None, save plots to this directory during bvps.
        """
        #Get info about IVP
        self.flow       = flow
        self.solver     = solver
        self.nx         = nx
        self.nz         = nz

        #Specify how BVPs work
        self.num_bvps           = num_bvps
        self.min_bvp_time       = min_bvp_time
        self.min_avg_dt         = min_avg_dt
        self.curr_avg_dt        = 0.
        self.completed_bvps     = 0
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = 0.
        self.bvp_equil_time     = bvp_equil_time
        self.bvp_transient_time = bvp_transient_time
        self.avg_started        = False
        self.final_equil_time   = final_equil_time

        # Stop parameters for bvps
        self.bvp_run_threshold      = bvp_run_threshold
        self.bvp_l2_check_time      = 1
        self.bvp_l2_last_check_time = 0
        self.do_bvp                 = False
        self.first_l2               = True

        #Get info about MPI distribution
        self.comm           = comm
        self.rank           = comm.rank
        self.size           = comm.size
        self.n_per_proc     = self.nz/self.size

        # Set up tracking dictionaries for flow fields
        for fd in self.FIELDS.keys():
            field, avg_type = self.FIELDS[fd]
            if avg_type == 0:
                self.flow.add_property('plane_avg({})'.format(field), name='{}'.format(fd))
            elif avg_type == 1:
                self.flow.add_property('plane_std({})'.format(field), name='{}'.format(fd))
            else:
                self.flow.add_property('{}'.format(field), name='{}'.format(fd))
                
        if self.rank == 0:
            self.profiles_dict = OrderedDict()
            self.profiles_dict_last, self.profiles_dict_curr = OrderedDict(), OrderedDict()
            for fd, info in self.FIELDS.items():
                if info[1] == 0 or info[1] == 1:    
                    self.profiles_dict[fd]      = np.zeros(nz)
                    self.profiles_dict_last[fd] = np.zeros(nz)
                    self.profiles_dict_curr[fd] = np.zeros(nz)
                else:   
                    self.profiles_dict[fd]      = np.zeros((nx,nz))
                    self.profiles_dict_last[fd] = np.zeros((nx,nz))
                    self.profiles_dict_curr[fd] = np.zeros((nx,nz))

        # Set up a dictionary of partial profiles to track averages locally so we
        # don't have to communicate each timestep.
        self.partial_prof_dict = OrderedDict()
        self.current_local_avg = OrderedDict()
        self.current_local_l2  = OrderedDict()
        for fd, info in self.FIELDS.items():
            if info[1] == 0 or info[1] == 1 or info[1] == 2:
                self.partial_prof_dict[fd]  = np.zeros(self.n_per_proc)
                self.current_local_avg[fd]  = np.zeros(self.n_per_proc)
                self.current_local_l2[fd]   = np.zeros(self.n_per_proc)
            else:
                self.partial_prof_dict[fd]  = np.zeros((nx, self.n_per_proc))
                self.current_local_avg[fd]  = np.zeros((nx, self.n_per_proc))
#            self.current_local_l2[fd] = 0

        # Set up a dictionary which tracks the states of important variables in the solver.
        self.solver_states = OrderedDict()
        self.vel_solver_states = OrderedDict()
        for st, var in self.VARS.items():
            self.solver_states[st] = self.solver.state[var]
        for st, var in self.VEL_VARS.items():
            self.vel_solver_states[st] = self.solver.state[var]


        self.plot_dir = plot_dir
        self.files_saved = 0
        if not isinstance(self.plot_dir, type(None)):
            import os
            if self.rank == 0 and not os.path.exists('{:s}'.format(self.plot_dir)):
                os.mkdir('{:s}'.format(self.plot_dir))


    def update_local_profiles(self, dt, prof_name, avg_type=0):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, 
        update the average on the local core based on the current flows.

        Arguments:
            dt              - size of timestep taken
            prof_name       - A string, which is a key to the class FIELDS dictionary
            avg_type        - If 0, horiz avg.  If 1, full 2D field.
        """
        if avg_type == 0:
            self.partial_prof_dict[prof_name] += \
                        dt*self.flow.properties['{}'.format(prof_name)]['g'][0,:]
        elif avg_type == 1:
            self.partial_prof_dict[prof_name] += \
                        dt*self.flow.properties['{}'.format(prof_name)]['g'][0,:]**2
        elif avg_type == 2:
            self.partial_prof_dict[prof_name] += \
                        dt*np.sum((self.flow.properties['{}'.format(prof_name)]['g'] - np.mean(self.flow.properties['{}'.format(prof_name)]['g'], axis=0))**2, axis=0)
        elif avg_type == 3:
            self.partial_prof_dict[prof_name] += \
                        dt*self.flow.properties['{}'.format(prof_name)]['g']

    def _update_profiles_dict(self, *args, **kwargs):
        pass

    def get_full_profile(self, prof_name, avg_type=0):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, communicate the
        full vertical profile across all processes, then return the full profile as a function
        of depth.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
            avg_type        - If 0, horiz avg.  If 1, full 2D field.
        """
        if avg_type == 0 or avg_type == 1 or avg_type == 2:
            local = np.zeros(self.nz)
            glob  = np.zeros(self.nz)
            local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.partial_prof_dict[prof_name]
        elif avg_type == 3:
            local = np.zeros((self.nx,self.nz))
            glob  = np.zeros((self.nx,self.nz))
            local[:,self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.partial_prof_dict[prof_name]
        self.comm.Allreduce(local, glob, op=MPI.SUM)

        profile = glob
        return profile

    def update_avgs(self, dt, min_Re = 1):
        """
        If proper conditions are met, this function adds the time-weighted vertical profile
        of all profiles in FIELDS to the appropriate arrays which are tracking classes. The
        size of the timestep is also recorded.

        The averages taken by this class are time-weighted averages of horizontal averages, such
        that sum(dt * Profile) / sum(dt) = <time averaged profile used for BVP>

        Arguments:
            dt          - The size of the current timestep taken.
            min_Re      - Only count this timestep toward the average if vol_avg(Re) is greater than this.
        """
        #Don't average if all BVPs are done
        if self.completed_bvps >= self.num_bvps:
            return

        if self.flow.grid_average('Re') > min_Re:
            if not self.avg_started:
                self.avg_started=True
                self.avg_time_start = self.solver.sim_time
            # Don't count point if a BVP has been completed very recently
            if self.completed_bvps == 0:
                if (self.solver.sim_time - self.avg_time_start) < self.bvp_transient_time:
                    return
            else:
                if (self.solver.sim_time - self.avg_time_start) < self.bvp_equil_time:
                    return

            #Update sums for averages. Check to see if we're converged enough for a BVP.
            self.avg_time_elapsed += dt
            self.curr_avg_dt      += dt
            if self.curr_avg_dt >= self.min_avg_dt:
                for fd, info in self.FIELDS.items():
                    field, avg_type = self.FIELDS[fd]
                    self.update_local_profiles(self.curr_avg_dt, fd, avg_type=avg_type)
                    avg = self.partial_prof_dict[fd]/self.avg_time_elapsed
                    if self.first_l2:
                        self.current_local_l2[fd]  *= 0
                        self.current_local_avg[fd] = avg
                        continue
                    else:
                        self.current_local_l2[fd] = np.abs((self.current_local_avg[fd] - avg)/self.current_local_avg[fd])
                        self.current_local_avg[fd] = avg

                if (self.solver.sim_time - self.bvp_l2_last_check_time) > self.bvp_l2_check_time and not self.first_l2:
                    local, globl = np.zeros(len(self.FIELDS.keys())), np.zeros(len(self.FIELDS.keys()))
                    for i, k in enumerate(self.FIELDS.keys()):
                        local[i] = np.max(self.current_local_l2[k])
                    self.comm.Allreduce(local, globl, op=MPI.MAX)
                    logger.info('Max l2 norm for convergence: {:.4g} / {:.4g} for BVP solve'.format(np.max(globl), self.bvp_run_threshold))
                    if np.max(globl) < self.bvp_run_threshold:
                        self.do_bvp = True
                    else:
                        self.do_bvp = False
                    self.bvp_l2_last_check_time = self.solver.sim_time
                self.curr_avg_dt = 0.
                self.first_l2 = False


    def check_if_solve(self):
        """ Returns a boolean.  If True, it's time to solve a BVP """
        return (self.avg_started and self.avg_time_elapsed >= self.min_bvp_time) and (self.do_bvp and (self.completed_bvps < self.num_bvps))

    def _save_file(self):
        """  Saves profiles dict to file """
        if not isinstance(self.plot_dir, type(None)):
            z_profile = np.zeros(self.nz)
            z_profile[self.rank*self.n_per_proc:(self.rank+1)*self.n_per_proc] = self.solver.domain.grid(-1)
            global_z = np.zeros_like(z_profile)
            self.comm.Allreduce(z_profile, global_z, op=MPI.SUM)
            if self.rank == 0:
                file_name = self.plot_dir + "profile_dict_file_{:04d}.h5".format(self.files_saved)
                with h5py.File(file_name, 'w') as f:
                    for k, item in self.profiles_dict.items():
                        f[k] = item
                    f['z'] = global_z
            self.files_saved += 1
                    
            

    def terminate_IVP(self):
        if not isinstance(self.final_equil_time, type(None)):
            if ((self.solver.sim_time - self.avg_time_start) >= self.final_equil_time) and (self.completed_bvps >= self.num_bvps):
                return True
        else:
            return False
            

    def _reset_fields(self):
        """ Reset all local fields after doing a BVP """
        self.do_bvp = False
        self.first_l2 = False
        # Reset profile arrays for getting the next bvp average
        for fd, info in self.FIELDS.items():
            if self.rank == 0:
                self.profiles_dict_last[fd] = self.profiles_dict_curr[fd]
                self.profiles_dict[fd]      *= 0
            self.partial_prof_dict[fd]  *= 0
            self.current_local_avg[fd]  *= 0
            self.current_local_l2[fd]  *= 0

    def _set_subs(self, problem):
        pass
    
    def _set_eqns(self, problem):
        pass

    def _set_BCs(self, problem):
        pass


    def solve_BVP(self):
        """ Base functionality at the beginning of BVP solves, regardless of equation set"""

        for fd, item in self.FIELDS.items():
            defn, avg_type = item
            curr_profile = self.get_full_profile(fd, avg_type=avg_type)
            if self.rank == 0:
                if item[1] == 0 or item[1] == 3:
                    self.profiles_dict[fd] = curr_profile/self.avg_time_elapsed
                else:
                    self.profiles_dict[fd] = np.sqrt(curr_profile/self.avg_time_elapsed)
                self.profiles_dict_curr[fd] = 1*self.profiles_dict[fd]
        self._save_file()

        # Restart counters for next BVP
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = self.solver.sim_time
        self.completed_bvps     += 1

class BoussinesqBVPSolver(BVPSolverBase):
    """
    Inherits the functionality of BVP_solver_base in order to solve BVPs involving
    the Boussinesq equations in the middle of time evolution of IVPs.

    Solves energy equation.  Makes no approximations other than time-stationary dynamics.
    """

    # 0 - full avg profile
    # 1 - stdev profile
    # 2 - stdev of full avg field
    # 3 - full avg field
    FIELDS = OrderedDict([  
                ('enth_flux_IVP',       ('w*(T0+T1)', 0)),                      
                ('tot_flux_IVP',        ('(w*(T0+T1) - P*(T0_z+T1_z))', 0)),                      
                        ])
    VARS   = OrderedDict([  
                ('T1_IVP',              'T1'),
                ('T1_z_IVP',            'T1_z'), 
                ('p_IVP',               'p'), 
                        ])
    VEL_VARS = OrderedDict([
                ('w_IVP',               'w'), 
                ('wz_IVP',              'wz'), 
                ('u_IVP',               'u'), 
                ('uz_IVP',              'uz'), 
                        ])

    def __init__(self, atmosphere_class, *args, **kwargs):
        self.atmosphere_class = atmosphere_class
        self.plot_count = 0
        super(BoussinesqBVPSolver, self).__init__(*args, **kwargs)
    
    def _set_eqns(self, problem):
        """ Sets the horizontally-averaged boussinesq equations """
        logger.debug('setting T1_z eqn')
        problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug('Setting energy equation')
        problem.add_equation(("P*dz(T1_z) = dz(enth_flux_IVP - P*T0_z)"))
        
        logger.debug('Setting HS equation')
        problem.add_equation(("dz(p1) - T1 = 0"))
        
    def _set_BCs(self, atmosphere, bc_kwargs):
        """ Sets standard thermal BCs, and also enforces the m = 0 pressure constraint """
        atmosphere.dirichlet_set = []
        if bc_kwargs['fixed_flux']:
            logger.info("Thermal BC: fixed_flux (BVP form)")
            atmosphere.problem.add_bc( "left(T1_z) = 0")
            atmosphere.problem.add_bc( "left(T1) + right(T1) = 0")
            atmosphere.dirichlet_set.append('T1')
            atmosphere.dirichlet_set.append('T1_z')
        else:
            atmosphere.set_thermal_BC(**bc_kwargs)
        atmosphere.problem.add_bc('right(p1) = 0')
        for key in atmosphere.dirichlet_set:
            atmosphere.problem.meta[key]['z']['dirichlet'] = True

    def _update_profiles_dict(self, bc_kwargs, atmosphere, vel_adjust_factor):
        """
        Update the enthalpy flux profile such that the BVP solve gets us the right answer.
        """

        #Get the atmospheric z-points (on the right size grid)
        z = atmosphere._new_field()
        z['g'] = atmosphere.z
        z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        z = z['g']

        #Keep track of initial flux profiles, then make a new enthalpy flux profile
        init_kappa_flux = 1*self.profiles_dict['tot_flux_IVP'] - self.profiles_dict['enth_flux_IVP']
        init_enth_flux = 1*self.profiles_dict['enth_flux_IVP']

        atmosphere.T0_z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        flux_through_system = -atmosphere.P * atmosphere.T0_z['g']

        #Scale flux appropriately.
        self.profiles_dict['enth_flux_IVP'] *= flux_through_system/self.profiles_dict['tot_flux_IVP']

        #Make some plots
        if not isinstance(self.plot_dir, type(None)):
            plt.plot(z, init_kappa_flux + init_enth_flux)
            plt.plot(z, init_kappa_flux)
            plt.plot(z, init_enth_flux)
            plt.plot(z, self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.plot(z, flux_through_system)
            plt.savefig('{}/fluxes_before_{:04d}.png'.format(self.plot_dir, self.plot_count))
            plt.close()
            print('flux ratio', self.profiles_dict['enth_flux_IVP']/init_enth_flux)
            plt.plot(z, self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.plot(z, flux_through_system - self.profiles_dict['enth_flux_IVP'], lw=2, ls='--')
            plt.savefig('{}/fluxes_after_{:04d}.png'.format(self.plot_dir, self.plot_count))
            plt.close()
            for fd in self.FIELDS.keys():
                if self.FIELDS[fd][1] == 0 or self.FIELDS[fd][1] == 1 or self.FIELDS[fd][1] == 2:
                    plt.plot(z, self.profiles_dict[fd])
                    plt.savefig('{}/{}_{:04d}.png'.format(self.plot_dir, fd, self.plot_count))
                    plt.close()
        self.plot_count += 1


    def solve_BVP(self, atmosphere_kwargs, diffusivity_args, bc_kwargs, tolerance=1e-10):
        """
        Solves a BVP in a 2D Boussinesq box.

        The BVP calculates updated temperature / pressure fields, then updates 
        the solver states which are tracked in self.solver_states.  
        This automatically updates the IVP's fields.

        """
        super(BoussinesqBVPSolver, self).solve_BVP()
        nz = atmosphere_kwargs['nz']
        # Create space for the returned profiles on all processes.
        return_dict = OrderedDict()
        for v in self.VARS.keys():
            return_dict[v] = np.zeros(self.nz, dtype=np.float64)


        # No need to waste processor power on multiple bvps, only do it on one
        if self.rank == 0:
            vel_adjust_factor = 1
            atmosphere = self.atmosphere_class(dimensions=1, comm=MPI.COMM_SELF, **atmosphere_kwargs)
            atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z','p1'], ncc_cutoff=tolerance)

            #Zero out old varables to make atmospheric substitutions happy.
            old_vars = ['u', 'w', 'dx(A)', 'uz', 'wz']
            for sub in old_vars:
                atmosphere.problem.substitutions[sub] = '0'

            atmosphere._set_parameters(*diffusivity_args)
            atmosphere._set_subs()

            # Create the appropriate enthalpy flux profile based on boundary conditions
            self._update_profiles_dict(bc_kwargs, atmosphere, vel_adjust_factor)

            #Add time and horizontally averaged profiles from IVP to the problem as parameters
            for k in self.FIELDS.keys():
                f = atmosphere._new_ncc()
                f.set_scales(self.nz / nz, keep_data=True) #If nz(bvp) =/= nz(ivp), this allows interaction between them
                if len(self.profiles_dict[k].shape) == 2:
                    f['g'] = self.profiles_dict[k].mean(axis=0)
                else:
                    f['g'] = self.profiles_dict[k]
                atmosphere.problem.parameters[k] = f

            self._set_eqns(atmosphere.problem)
            self._set_BCs(atmosphere, bc_kwargs)

            # Solve the BVP
            solver = atmosphere.problem.build_solver()

            pert = solver.perturbations.data
            pert.fill(1+tolerance)
            while np.sum(np.abs(pert)) > tolerance:
                solver.newton_iteration()
                logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))


            T1 = solver.state['T1']
            T1_z = solver.state['T1_z']
            P1   = solver.state['p1']

            #Appropriately adjust T1 in IVP
            T1.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] += T1['g']

            #Appropriately adjust T1_z in IVP
            T1_z.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_z_IVP'] += T1_z['g']

            #Appropriately adjust p in IVP
            P1.set_scales(self.nz/nz, keep_data=True)
            return_dict['p_IVP'] += P1['g']
        else:
            for v in self.VARS.keys():
                return_dict[v] *= 0
        logger.info(return_dict)
            
        self.comm.Barrier()
        # Communicate output profiles from proc 0 to all others.
        for v in self.VARS.keys():
            glob = np.zeros(self.nz)
            self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
            return_dict[v] = glob

        vel_adj_loc = np.zeros(1)
        vel_adj_glob = np.zeros(1)
        if self.rank == 0:
            vel_adj_loc[0] = vel_adjust_factor
        self.comm.Allreduce(vel_adj_loc, vel_adj_glob, op=MPI.SUM)

        # Actually update IVP states
        for v in self.VARS.keys():
            #Subtract out current avg
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] -= self.solver_states[v]['g'].mean(axis=0)
            #Put in right avg
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += return_dict[v][self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)]
        for v in self.VEL_VARS.keys():
            self.vel_solver_states[v]['g'] *= vel_adj_glob[0]

        self._reset_fields()


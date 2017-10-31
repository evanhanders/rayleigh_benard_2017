from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dedalus import public as de


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
        FIELDS - An OrderedDict of strings of which the time- and horizontally- averaged 
                 profiles are tracked (and fed into the BVP)
        VARS   - An OrderedDict of variables which will be updated by the BVP

    Object Attributes:
    ------------------
        avg_started         - If True, time averages for FIELDS has begun
        avg_time_elapsed    - Amount of IVP simulation time over which averages have been taken so far
        avg_time_start      - Simulation time at which average began
        bvp_equil_time      - Amount of sim time to wait for velocities to converge before starting averages
                                at the beginning of IVP or after a BVP is solved
        bvp_time            - Length of sim time to average over before doing bvp
        comm                - COMM_WORLD for IVP
        completed_bvps      - # of BVPs that have been completed during this run
        flow                - A dedalus flow_tools.GlobalFlowProperty object for the IVP solver which is tracking
                                the Reynolds number, and will track FIELDS variables
        n_per_proc          - Number of z-points per core (for parallelization)
        num_bvps            - Total number of BVPs to complete
        nz                  - z-resolution of the IVP grid
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

    def __init__(self, nz, flow, comm, solver, bvp_time, num_bvps, bvp_equil_time):
        """
        Initializes the object; grabs solver states and makes room for profile averages
        
        Arguments:
        nz              - the vertical resolution of the IVP
        flow            - a dedalus.extras.flow_tools.GlobalFlowProperty for the IVP solver
        comm            - An MPI comm object for the IVP solver
        solver          - The IVP solver
        bvp_time        - How often to perform a BVP, in sim time units
        num_bvps        - Maximum number of BVPs to solve
        bvp_equil_time  - Sim time to wait after a bvp before beginning averages for the next one
        """
        #Get info about IVP
        self.flow       = flow
        self.solver     = solver
        self.nz         = nz

        #Specify how BVPs work
        self.bvp_time           = bvp_time
        self.num_bvps           = num_bvps
        self.completed_bvps     = 0
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = 0.
        self.bvp_equil_time     = bvp_equil_time
        self.avg_started        = False

        #Get info about MPI distribution
        self.comm           = comm
        self.rank           = comm.rank
        self.size           = comm.size
        self.n_per_proc     = self.nz/self.size

        # Set up tracking dictionaries for flow fields
        for fd in self.FIELDS.keys():
            self.flow.add_property('plane_avg({})'.format(self.FIELDS[fd]), name='{}_avg'.format(fd))
        if self.rank == 0:
            self.profiles_dict = dict()
            self.profiles_dict_last, self.profiles_dict_curr = dict(), dict()
            for fd in self.FIELDS.keys():
                self.profiles_dict[fd]      = np.zeros(nz)
                self.profiles_dict_last[fd] = np.zeros(nz)
                self.profiles_dict_curr[fd] = np.zeros(nz)

        self.solver_states = dict()
        for st in self.VARS.keys():
            self.solver_states[st] = self.solver.state[self.VARS[st]]

    def get_full_profile(self, prof_name):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, communicate the
        full vertical profile across all processes, then return the full profile as a function
        of depth.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
        """
        local = np.zeros(self.nz)
        glob  = np.zeros(self.nz)
        local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.flow.properties['{}_avg'.format(prof_name)]['g'][0,:]
        self.comm.Allreduce(local, glob, op=MPI.SUM)
        return glob

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
            if (self.solver.sim_time - self.avg_time_start) < self.bvp_equil_time:
                return

            #Update sums for averages
            self.avg_time_elapsed += dt
            for fd in self.FIELDS.keys():
                curr_profile = self.get_full_profile(fd)
                if self.rank == 0:
                    self.profiles_dict[fd] += dt*curr_profile

    def check_if_solve(self):
        """ Returns a boolean.  If True, it's time to solve a BVP """
        return self.avg_started*(self.avg_time_elapsed >= self.bvp_time)*(self.completed_bvps < self.num_bvps)

    def _reset_fields(self):
        if self.rank != 0:
            return
        # Reset profile arrays for getting the next bvp average
        for fd in self.FIELDS.keys():
            self.profiles_dict[fd] = np.zeros(self.nz)

    def _set_subs(self, problem):
        pass
    
    def _set_eqns(self, problem):
        pass

    def _set_BCs(self, problem):
        pass


    def solve_BVP(self):
        """ Base functionality at the beginning of BVP solves, regardless of equation set"""

        for k in self.FIELDS.keys():
            if self.rank == 0:
                self.profiles_dict[k] /= self.avg_time_elapsed
                self.profiles_dict_curr[k] = 1*self.profiles_dict[k]

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

    FIELDS = OrderedDict([  
                ('T1_IVP',              'T1'),                      
                ('T1_z_IVP',            'T1_z'),                    
                ('UdotGrad_T',          'UdotGrad((T0+T1), (T0_z + T1_z))'),
                ('w_IVP',               'w'),
                ('Lap_T_IVP',           'Lap((T0+T1), (T0_z+T1_z))'),
                        ])
    VARS   = OrderedDict([  
                ('T1_IVP',              'T1'),
                ('T1_z_IVP',            'T1_z'), 
                        ])





    def __init__(self, atmosphere_class, *args, **kwargs):
        self.atmosphere_class = atmosphere_class
        super(BoussinesqBVPSolver, self).__init__(*args, **kwargs)
    
    def _set_subs(self, problem):
        # Full background thermodynamics from IVP
        problem.substitutions['T0_tot'] = '(T0 + T1_IVP)'
        problem.substitutions['T0_z_tot'] = '(T0_z + T1_z_IVP)'
        problem.substitutions['T0_zz_tot'] = '(T0_zz + dz(T1_z_IVP))'
        
    def _set_eqns(self, problem):
        logger.debug('setting T1_z eqn')
        problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug('Setting energy equation')
        problem.add_equation(("w_IVP * T1_z - P * dz(T1_z) = "
                                         "-UdotGrad_T + P*Lap_T_IVP"))
    def _set_BCs(self, problem):
        # Use thermal BCs from IVP
        problem.add_bc("left(T1_z) = 0") 
        problem.add_bc("right(T1) = 0")


    def solve_BVP(self, atmosphere_kwargs, diffusivity_args, tolerance=1e-13):
        """
        Solves a BVP in a 2D FC_ConstHeating atmosphere under the kappa/mu formulation of the equations.
        Run parameters are specified, and should be similar to those of the bvp.

        The BVP calculates updated rho / temperature fields, then updates the solver states which are
        tracked in self.solver_states.  This automatically updates the IVP's fields.

        """
        super(BoussinesqBVPSolver, self).solve_BVP()
        nz = atmosphere_kwargs['nz']

        # No need to waste processor power on multiple bvps, only do it on one
        if self.rank == 0:
            atmosphere = self.atmosphere_class(dimensions=1, comm=MPI.COMM_SELF, **atmosphere_kwargs)
            #Variables are T, dz(T), rho, integrated mass
            atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z',], ncc_cutoff=tolerance)

            #Zero out old varables to make atmospheric substitutions happy.
            old_vars = ['u', 'w', 'u_z', 'w_z', 'dx(A)']
            for sub in old_vars:
                atmosphere.problem.substitutions[sub] = '0'

            atmosphere._set_parameters(*diffusivity_args)
            atmosphere._set_subs()

            #Add time and horizontally averaged profiles from IVP to the problem as parameters
            for k in self.FIELDS.keys():
                f = atmosphere._new_ncc()
                f.set_scales(self.nz / nz, keep_data=True) #If nz(bvp) =/= nz(ivp), this allows interaction between them
                f['g'] = self.profiles_dict[k]
                atmosphere.problem.parameters[k] = f

            self._set_subs(atmosphere.problem)
            self._set_eqns(atmosphere.problem)
            self._set_BCs(atmosphere.problem)

            # Solve the BVP
            solver = atmosphere.problem.build_solver()

            pert = solver.perturbations.data
            pert.fill(1+tolerance)
            while np.sum(np.abs(pert)) > tolerance:
                solver.newton_iteration()
                logger.info('Perturbation norm: {}'.format(np.sum(np.abs(pert))))

            T1 = solver.state['T1']
            T1_z = solver.state['T1_z']

        # Create space for the returned profiles on all processes.
        return_dict = dict()
        for v in self.VARS.keys():
            return_dict[v] = np.zeros(self.nz, dtype=np.float64)

        if self.rank == 0:
            #Appropriately adjust T1 in IVP
            T1.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_IVP'] = T1['g'] + self.profiles_dict['T1_IVP'] - self.profiles_dict_curr['T1_IVP']

            #Appropriately adjust T1_z in IVP
            T1_z.set_scales(self.nz/nz, keep_data=True)
            return_dict['T1_z_IVP'] = T1_z['g'] + self.profiles_dict['T1_z_IVP'] - self.profiles_dict_curr['T1_z_IVP']
        print(return_dict)

        self.comm.Barrier()
        # Communicate output profiles from proc 0 to all others.
        for v in self.VARS.keys():
            glob = np.zeros(self.nz)
            self.comm.Allreduce(return_dict[v], glob, op=MPI.SUM)
            return_dict[v] = glob

        # Actually update IVP states
        for v in self.VARS.keys():
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += return_dict[v][self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)]

        self._reset_fields()


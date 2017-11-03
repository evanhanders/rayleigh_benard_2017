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
        bvp_transient_time  - Amount of time to wait at the beginning of the sim
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
    VEL_VARS   = None

    def __init__(self, nx, nz, flow, comm, solver, bvp_time, num_bvps, bvp_equil_time, bvp_transient_time=0,
                 bvp_pairs=False):
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
        self.nx         = nx
        self.nz         = nz

        #Specify how BVPs work
        self.bvp_time           = bvp_time
        self.num_bvps           = num_bvps
        self.completed_bvps     = 0
        self.avg_time_elapsed   = 0.
        self.avg_time_start     = 0.
        self.bvp_equil_time     = bvp_equil_time
        self.bvp_transient_time = bvp_transient_time
        self.avg_started        = False
        self.bvp_pairs          = bvp_pairs

        #Get info about MPI distribution
        self.comm           = comm
        self.rank           = comm.rank
        self.size           = comm.size
        self.n_per_proc     = self.nz/self.size

        # Set up tracking dictionaries for flow fields
        added_fields = []
        for fd in self.FIELDS.keys():
            field, avg_type = self.FIELDS[fd]
            if avg_type == 0:
                self.flow.add_property('plane_avg({})'.format(field), name='{}'.format(fd))
            else:
                self.flow.add_property('{}'.format(field), name='{}'.format(fd))
                
        if self.rank == 0:
            self.profiles_dict = dict()
            self.profiles_dict_last, self.profiles_dict_curr = dict(), dict()
            for fd, info in self.FIELDS.items():
                if info[1] == 0:    
                    self.profiles_dict[fd]      = np.zeros(nz)
                    self.profiles_dict_last[fd] = np.zeros(nz)
                    self.profiles_dict_curr[fd] = np.zeros(nz)
                else:   
                    self.profiles_dict[fd]      = np.zeros((nx,nz))
                    self.profiles_dict_last[fd] = np.zeros((nx,nz))
                    self.profiles_dict_curr[fd] = np.zeros((nx,nz))

        self.solver_states = dict()
        self.vel_solver_states = dict()
        for st, var in self.VARS.items():
            self.solver_states[st] = self.solver.state[var]
        for st, var in self.VEL_VARS.items():
            self.vel_solver_states[st] = self.solver.state[var]

    def get_full_profile(self, prof_name, avg_type=0):
        """
        Given a profile name, which is a key to the class FIELDS dictionary, communicate the
        full vertical profile across all processes, then return the full profile as a function
        of depth.

        Arguments:
            prof_name       - A string, which is a key to the class FIELDS dictionary
        """
        if avg_type == 0:
            local = np.zeros(self.nz)
            glob  = np.zeros(self.nz)
            local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.flow.properties['{}'.format(prof_name)]['g'][0,:]
        elif avg_type == 1:
            local = np.zeros((self.nx,self.nz))
            glob  = np.zeros((self.nx,self.nz))
            local[:,self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.flow.properties['{}'.format(prof_name)]['g']
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
            if self.completed_bvps == 0:
                if (self.solver.sim_time - self.avg_time_start) < self.bvp_transient_time:
                    return
            else:
                if (self.solver.sim_time - self.avg_time_start) < self.bvp_equil_time:
                    return

            #Update sums for averages
            self.avg_time_elapsed += dt
            for fd in self.FIELDS.keys():
                field, avg_type = self.FIELDS[fd]
                curr_profile = self.get_full_profile(fd, avg_type=avg_type)
                if self.rank == 0:
                    self.profiles_dict[fd] += dt*curr_profile

    def check_if_solve(self):
        """ Returns a boolean.  If True, it's time to solve a BVP """
        return self.avg_started*(self.avg_time_elapsed >= self.bvp_time)*(self.completed_bvps < self.num_bvps)

    def _reset_fields(self):
        if self.rank != 0:
            return
        # Reset profile arrays for getting the next bvp average
        for fd, info in self.FIELDS.items():
#            if self.completed_bvps == 1:
            self.profiles_dict_last[fd] = self.profiles_dict_curr[fd]
#            else:
#                self.profiles_dict_last[fd] += self.profiles_dict_curr[fd]
#                self.profiles_dict_last[fd] /= 2.
            if info[1] == 0:
                self.profiles_dict[fd] = np.zeros(self.nz)
            else:
                self.profiles_dict[fd] = np.zeros((self.nx, self.nz))

    def _set_subs(self, problem):
        pass
    
    def _set_eqns(self, problem):
        pass

    def _set_BCs(self, problem):
        pass


    def solve_BVP(self):
        """ Base functionality at the beginning of BVP solves, regardless of equation set"""

        keys = list(self.FIELDS.keys())
        for k in keys:
            if self.rank == 0:
                self.profiles_dict[k] /= self.avg_time_elapsed
                self.profiles_dict_curr[k] = 1*self.profiles_dict[k]

                if self.completed_bvps % 2 != 0 and self.bvp_pairs:
                    self.profiles_dict[k] += self.profiles_dict_last[k]
                    self.profiles_dict[k] /= 2.

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
    # 1 - full avg field
    FIELDS = OrderedDict([  
                ('T1_IVP',              ('T1', 0)),                      
                ('T1_z_IVP',            ('T1_z', 0)),                    
                ('T1_IVP_full',         ('T1', 1)),                      
                ('T1_z_IVP_full',       ('T1_z', 1)),                    
                ('T1_zz_IVP_full',      ('dz(T1_z)', 1)),                    
                ('w_IVP_full',          ('w', 1)),                    
                ('wz_IVP_full',          ('wz', 1)),                    
                ('u_IVP_full',          ('u', 1)),                    
                ('p_IVP',               ('p', 0)), 
                ('T_forcing',           ('(UdotGrad((T0+T1), (T0_z+T1_z)) - dz(P * T1_z))', 0)),
#                ('T_forcing',           ('dz(w*(T1) - P * T1_z)', 0)),
                ('Lap_w',               ('Lap(w, wz)', 0)),
                ('UdotGrad_w',          ('UdotGrad(w, wz)', 0)),
                ('w_forcing',           ('(-UdotGrad(w, wz) - dz(p) + T1 + R*Lap(w, wz))', 0)),
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
        super(BoussinesqBVPSolver, self).__init__(*args, **kwargs)
    
    def _set_eqns(self, problem):
        """ Sets the horizontally-averaged boussinesq equations """
        logger.debug('setting T1_z eqn')
        problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug('Setting energy equation')
        problem.add_equation(("P*dz(T1_z) = T_forcing"))
        
        logger.debug('Setting HS equation')
        problem.add_equation(("dz(p1) - T1 =  w_forcing"))
        
    def _set_BCs(self, atmosphere, bc_kwargs):
        """ Sets standard thermal BCs, and also enforces the m = 0 pressure constraint """
        atmosphere.dirichlet_set = []
        atmosphere.set_thermal_BC(**bc_kwargs)
        atmosphere.problem.add_bc('right(p1) = 0')
        for key in atmosphere.dirichlet_set:
            atmosphere.problem.meta[key]['z']['dirichlet'] = True

    def solve_BVP(self, atmosphere_kwargs, diffusivity_args, bc_kwargs, tolerance=1e-13):
        """
        Solves a BVP in a 2D Boussinesq box.

        The BVP calculates updated temperature / pressure fields, then updates 
        the solver states which are tracked in self.solver_states.  
        This automatically updates the IVP's fields.

        """
        super(BoussinesqBVPSolver, self).solve_BVP()
        nz = atmosphere_kwargs['nz']
        # Create space for the returned profiles on all processes.
        return_dict = dict()
        for v in self.VARS.keys():
            return_dict[v] = np.zeros(self.nz, dtype=np.float64)
#                return_dict[v] -= self.get_full_profile(v)


        # No need to waste processor power on multiple bvps, only do it on one
        if self.rank == 0:
            avg_change = 1e10
            vel_adjust_factor = 1            
            while avg_change > 1e-9:
                atmosphere = self.atmosphere_class(dimensions=1, comm=MPI.COMM_SELF, **atmosphere_kwargs)
                atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z','p1'], ncc_cutoff=tolerance)

                #Zero out old varables to make atmospheric substitutions happy.
                old_vars = ['u', 'w', 'u_z', 'w_z', 'dx(A)']
                for sub in old_vars:
                    atmosphere.problem.substitutions[sub] = '0'

                atmosphere._set_parameters(*diffusivity_args)
                atmosphere._set_subs()
                keys = list(self.FIELDS.keys())
#                for k in keys:
#                    f = atmosphere._new_ncc()
#                    f['g'] = atmosphere.z
#                    f.set_scales(self.nz/nz, keep_data=True)
#                    plt.plot(f['g'], self.profiles_dict[k])
#                    plt.savefig('{}_{}.png'.format(k, self.completed_bvps))
#                    plt.close()
                
    #            T1 = atmosphere._new_field()
    #            T1_z = atmosphere._new_field()
    #            T1_zz = atmosphere._new_field()
    #
    #            delta_t = atmosphere.thermal_time*(1 - np.exp(-0.05))
    #            T1.set_scales(self.nz/nz, keep_data=False)
    #            T1['g'] = self.profiles_dict['T_forcing']*delta_t
    #            T1_z.set_scales(self.nz/nz, keep_data=False)
    #            T1_z['g'] = self.profiles_dict['T_z_forcing']*delta_t
    #            T1.differentiate('z', out=T1_z)

    #            T1_zz.set_scales(self.nz/nz, keep_data=False)
    #            T1_zz['g'] = self.profiles_dict['T_zz_forcing']*atmosphere.thermal_time*(1 - np.exp(-0.2))
    #            T1_zz.antidifferentiate('z', ('left', 0), out=T1_z)
    #            T1_z.antidifferentiate('z', ('right', 0), out=T1)

                #Add time and horizontally averaged profiles from IVP to the problem as parameters
                for k in keys:
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
                avg_change = np.mean(np.abs(T1['g']))
                logger.info('avg change: {}'.format(avg_change))
                logger.info('T1 change: {}'.format(T1['g']))
                T1.set_scales(self.nz/nz, keep_data=True)
                self.profiles_dict['T1_IVP_full'] += T1['g']
                T1_z = solver.state['T1_z']
                T1_z.set_scales(self.nz/nz, keep_data=True)
                self.profiles_dict['T1_z_IVP_full'] += T1_z['g']
                T1_zz = atmosphere._new_field()
                T1_z.differentiate('z', out=T1_zz)
                T1_zz.set_scales(self.nz/nz, keep_data=True)
                self.profiles_dict['T1_zz_IVP_full'] += T1_zz['g']

                P1   = solver.state['p1']
                P1.set_scales(self.nz/nz, keep_data=True)
                self.profiles_dict['p_IVP'] += P1['g']
                P1_z = atmosphere._new_field()
                P1.differentiate('z', out=P1_z)
                P1_z.set_scales(self.nz/nz, keep_data=True)

                
                self.profiles_dict['w_forcing'] = \
                    (-self.profiles_dict['UdotGrad_w'] - P1_z['g'] \
                     + self.profiles_dict['T1_IVP_full'] \
                     + atmosphere.R*self.profiles_dict['Lap_w']).mean(axis=0)
                atmosphere.T0_z.set_scales(self.nz/nz, keep_data=True)
                atmosphere.T0.set_scales(self.nz/nz, keep_data=True)
                self.profiles_dict['T_forcing'] = \
                    (self.profiles_dict['w_IVP_full']\
                     *(self.profiles_dict['T1_z_IVP_full']+atmosphere.T0_z['g'])\
                     + self.profiles_dict['wz_IVP_full'] * \
                      (self.profiles_dict['T1_IVP_full'] + atmosphere.T0['g'])\
                     - atmosphere.P*self.profiles_dict['T1_zz_IVP_full']).mean(axis=0)

                atmosphere.T0.set_scales(self.nz/nz, keep_data=True)
                enth_flux = (self.profiles_dict['w_IVP_full']*(self.profiles_dict['T1_IVP_full']+atmosphere.T0['g'])).mean(axis=0)
                mid_enth_flux = enth_flux[int(len(enth_flux)/2)]
                vel_adjust = atmosphere.P / mid_enth_flux
                vel_adjust_factor *= vel_adjust
                self.profiles_dict['w_IVP_full'] *= vel_adjust
                self.profiles_dict['wz_IVP_full'] *= vel_adjust

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
        print(return_dict)
            
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
            self.solver_states[v].set_scales(1, keep_data=True)
            self.solver_states[v]['g'] += return_dict[v][self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)]
        for v in self.VEL_VARS.keys():
            self.vel_solver_states[v]['g'] *= vel_adj_glob[0]

        self._reset_fields()


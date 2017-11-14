from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)


from mpi4py import MPI
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
                 bvp_run_threshold=1e-2, bvp_l2_check_time=1, min_bvp_time=50, plot_dir=None,
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
            else:
                self.flow.add_property('{}'.format(field), name='{}'.format(fd))
                
        if self.rank == 0:
            self.profiles_dict = OrderedDict()
            self.profiles_dict_last, self.profiles_dict_curr = OrderedDict(), OrderedDict()
            for fd, info in self.FIELDS.items():
                if info[1] == 0:    
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
            if info[1] == 0:
                self.partial_prof_dict[fd]  = np.zeros(self.n_per_proc)
                self.current_local_avg[fd]  = np.zeros(self.n_per_proc)
            else:
                self.partial_prof_dict[fd]  = np.zeros((nx, self.n_per_proc))
                self.current_local_avg[fd]  = np.zeros((nx, self.n_per_proc))
            self.current_local_l2[fd] = 0

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
        if avg_type == 0:
            local = np.zeros(self.nz)
            glob  = np.zeros(self.nz)
            local[self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.partial_prof_dict[prof_name]
        elif avg_type == 1:
            local = np.zeros((self.nx,self.nz))
            glob  = np.zeros((self.nx,self.nz))
            local[:,self.n_per_proc*self.rank:self.n_per_proc*(self.rank+1)] = \
                        self.partial_prof_dict[prof_name]
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

            #Update sums for averages. Check to see if we're converged enough for a BVP.
            self.avg_time_elapsed += dt
            self.curr_avg_dt      += dt
            if self.curr_avg_dt >= self.min_avg_dt:
                for fd, info in self.FIELDS.items():
                    field, avg_type = self.FIELDS[fd]
                    self.update_local_profiles(self.curr_avg_dt, fd, avg_type=avg_type)
                    avg = self.partial_prof_dict[fd]/self.avg_time_elapsed
                    if self.first_l2:
                        self.current_local_l2[fd]  = 0
                        self.current_local_avg[fd] = avg
                        continue
                    else:
                        self.current_local_l2[fd] = np.sum(np.abs((self.current_local_avg[fd] - avg)/self.current_local_avg[fd]))
                        self.current_local_avg[fd] = avg

                if (self.solver.sim_time - self.bvp_l2_last_check_time) > self.bvp_l2_check_time and not self.first_l2:
                    local, globl = np.zeros(len(self.FIELDS.keys())), np.zeros(len(self.FIELDS.keys()))
                    for i, k in enumerate(self.FIELDS.keys()):
                        local[i] = self.current_local_l2[k]
                    self.comm.Allreduce(local, globl, op=MPI.SUM)
                    globl /= self.nz
                    logger.info('Max avg convergence: {:.4g} / {:.4g} for BVP solve'.format(np.max(globl), self.bvp_run_threshold))
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
                        print(k, item)
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
                self.profiles_dict[fd] = curr_profile/self.avg_time_elapsed
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
    # 1 - full avg field
    FIELDS = OrderedDict([  
                ('enth_flux_IVP',       ('w*(T0+T1)', 0)),                      
                ('T_z_IVP',             ('(T0_z+T1_z)', 0)),                      
                ('T1_IVP',              ('T1', 0)),                      
                ('T1_z_IVP',            ('T1_z', 0)),                    
                ('p_IVP',               ('p', 0)), 
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
        problem.add_equation(("P*dz(T1_z) = dz(enth_flux_IVP - P*T_z_IVP)"))
        
        logger.debug('Setting HS equation')
        problem.add_equation(("dz(p1) - T1 = 0"))# w_forcing"))
        
    def _set_BCs(self, atmosphere, bc_kwargs):
        """ Sets standard thermal BCs, and also enforces the m = 0 pressure constraint """
        atmosphere.dirichlet_set = []
        atmosphere.set_thermal_BC(**bc_kwargs)
        atmosphere.problem.add_bc('right(p1) = 0')
        for key in atmosphere.dirichlet_set:
            atmosphere.problem.meta[key]['z']['dirichlet'] = True

    def _find_BL_thick(self, z, atmosphere):
        """
        Find boundary layer thickness of top and bottom temperature profiles
        """
        atmosphere.T0.set_scales(self.nz/atmosphere.nz, keep_data=True)
        T_profile = self.profiles_dict['T1_IVP'] + atmosphere.T0['g']

        mid = int(len(z)/2)

        n_pts_start = 3

        start_pct = 10
        n  = int(len(z)/start_pct)
        xs = z[mid-n:mid+n]
        ys = T_profile[mid-n:mid+n]
        powers = np.arange(5)
        for i in range(len(powers)):
            n = 1+i
            p = np.polyfit(xs, ys, n)
            line = np.zeros_like(xs)
            for j in range(n+1):
                line += p[j]*xs**(n-j)
            powers[i] = np.sum(np.abs( (line - ys) / ys))

        pow_fit = np.argmin(powers) + 1
        p  = np.polyfit(xs, ys, pow_fit)
        line = np.zeros_like(z)
        for i in range(pow_fit+1):
            line += p[i]*z**(pow_fit-i)

        last = np.where(np.abs((T_profile - line)/line) < 0.1)[0][-1]
        first = np.where(np.abs((T_profile - line)/line) < 0.1)[0][0]
        print(first, last, T_profile, line, np.abs((T_profile - line)/line))

       

#        # Fit to middle of domain
#        for i in range(len(z) - n_pts_start):
#            mid_z = z[i:n_pts_start+i]
#            mid_prof = T_profile[i:n_pts_start+i]
#            p = np.polyfit(mid_z, mid_prof, 1)
#            slopes[i] = p[0]
#            inters[i] = p[1]
#        
#        #m_slopes = np.median(slopes[:int(len(slopes)/2)])
#        m_slopes = np.mean(slopes[2*len(slopes)/5:3*len(slopes)/5])
#        print(slopes, m_slopes, np.abs((slopes/m_slopes - 1)) < 0.5)
##        last     = np.where(np.abs((slopes/m_slopes - 1)) < 0.5)[0][-1] - mid
#        last     = np.where(np.abs(slopes) < 0.1)[0][-1] - mid
##        print(slopes, m_slopes, np.abs((slopes/m_slopes - 1)) < 0.5, last)
#
        xs = z[first:last]
        ys = T_profile[first:last]
        powers = np.arange(5)
        for i in range(len(powers)):
            n = 1+i
            p = np.polyfit(xs, ys, n)
            line = np.zeros_like(xs)
            for j in range(n+1):
                line += p[j]*xs**(n-j)
            powers[i] = np.sum(np.abs( (line - ys) / ys))

        pow_fit = np.argmin(powers) + 1
        p  = np.polyfit(xs, ys, pow_fit)
        line = np.zeros_like(z)
        for i in range(pow_fit+1):
            line += p[i]*z**(pow_fit-i)


        slopes_bot = np.zeros(first-n_pts_start)
        for i in range(first-n_pts_start):
            bot_z = z[:n_pts_start+i]
            bot_prof = T_profile[:n_pts_start+i]
            p = np.polyfit(bot_z, bot_prof, 1)
            slopes_bot[i] = p[0]
        
        m_slopes = np.mean(slopes_bot[:len(slopes_bot)/5])
        print(slopes_bot, m_slopes)
        last_bot     = np.where(np.abs((slopes_bot/m_slopes - 1)) < 0.05)[0][-1]
        fit_bot  = np.polyfit(z[:n_pts_start+last_bot], T_profile[:n_pts_start+last_bot], 1)


        slopes_top = np.zeros(len(z) - last - n_pts_start)
        for i in range(len(z) - last - n_pts_start):
            top_z = z[-n_pts_start-i:]
            top_prof = T_profile[-n_pts_start-i:]
            p = np.polyfit(top_z, top_prof, 1)
            slopes_top[i] = p[0]
        
        m_slopes = np.mean(slopes_top[:len(slopes_top)/5])
        last_top     = np.where(np.abs((slopes_top/m_slopes - 1)) < 0.05)[0][-1]
        fit_top  = np.polyfit(z[-n_pts_start-last_top:], T_profile[-n_pts_start-last_top:], 1)

        # Calculate boundary layer thicknesses
#        low_bl = (fit_bot[1] - fit_mid[1])/(fit_mid[0] - fit_bot[0])
#        upp_bl = (fit_top[1] - fit_mid[1])/(fit_mid[0] - fit_top[0])

        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        bot_func = interp1d(z, line - (fit_bot[0]*z + fit_bot[1]))
        top_func = interp1d(z, line - (fit_top[0]*z + fit_top[1]))
        low_bl = brentq(bot_func, np.min(z), 0.5*atmosphere.Lz)
        upp_bl = brentq(top_func, 0.5*atmosphere.Lz, np.max(z))

        # Make plot of boundary layer find
        plt.plot(z, T_profile)
        plt.plot(z, line)
        plt.plot(z, z*fit_bot[0] + fit_bot[1])
        plt.plot(z, z*fit_top[0] + fit_top[1])
        plt.plot(z[first:last],T_profile[first:last],ls='--', lw=2)
        plt.axvline(low_bl)
        plt.axvline(upp_bl)
        plt.ylim(np.min(T_profile), np.max(T_profile))
        plt.xlim(np.min(z), np.max(z))
        plt.savefig('{}/bl_find_{:04d}.png'.format(self.plot_dir, self.plot_count))
        plt.close()


        return low_bl, upp_bl, np.where(z > low_bl)[0][0], np.where(z < upp_bl)[0][-1]

#        bottom = z[0]
#        top    = z[-1]
#        quarter = bottom + 0.3*(top-bottom)
#        three_quarter = bottom + 0.7*(top-bottom)
#        xs = z[(z>=quarter)*(z<=three_quarter)]
#        ys = self.profiles_dict['T1_z_IVP'][(z>=quarter)*(z<=three_quarter)]
#
#        mean = np.mean(ys)
#        stdev = np.std(ys)
#
#        err = np.abs(self.profiles_dict['T1_z_IVP'] - mean)
#
#        # Find which points are in / out of boundary layers
#        lower = z[:int(len(z)/2)]
#        lower_T = self.profiles_dict['T1_IVP'][:int(len(z)/2)]
#        lower_e = err[:int(len(z)/2)]
#        upper = z[int(len(z)/2):]
#        upper_T = self.profiles_dict['T1_IVP'][int(len(z)/2):]
#        upper_e = err[int(len(z)/2):]
#
#        diff_bad = 3*stdev
#
#        lower_z = lower[lower_e > diff_bad]
#        lower_y = lower_T[lower_e > diff_bad]
#        upper_z = upper[upper_e > diff_bad]
#        upper_y = upper_T[upper_e > diff_bad]
#        
#        mid_lower_z = lower[lower_e < diff_bad]
#        mid_lower_y = lower_T[lower_e < diff_bad]
#        mid_upper_z = upper[upper_e < diff_bad]
#        mid_upper_y = upper_T[upper_e < diff_bad]
#
#        # Fit lines
#        line_low = np.polyfit(lower_z, lower_y, 1)
#        line_upp = np.polyfit(upper_z, upper_y, 1)
#        line_mid_low = np.polyfit(mid_lower_z, mid_lower_y, 1)
#        line_mid_upp = np.polyfit(mid_upper_z, mid_upper_y, 1)
#
#        # Calculate boundary layer thicknesses
#        low_bl = (line_low[1] - line_mid_low[1])/(line_mid_low[0] - line_low[0])
#        upp_bl = (line_upp[1] - line_mid_upp[1])/(line_mid_upp[0] - line_upp[0])
#
#        # Make plot of boundary layer find
#        plt.plot(z, self.profiles_dict['T1_IVP'])
#        plt.plot(lower, lower*line_mid_low[0] + line_mid_low[1])
#        plt.plot(upper, upper*line_mid_upp[0] + line_mid_upp[1])
#        plt.plot(z, z*line_low[0] + line_low[1])
#        plt.plot(z, z*line_upp[0] + line_upp[1])
#        plt.axvline(low_bl)
#        plt.axvline(upp_bl)
#        plt.ylim(np.min(self.profiles_dict['T1_IVP']), np.max(self.profiles_dict['T1_IVP']))
#        plt.xlim(np.min(z), np.max(z))
#        plt.savefig('{}/bl_find_{:04d}.png'.format(self.plot_dir, self.plot_count))
#        plt.close()
#
#        return low_bl, upp_bl, np.where(z > low_bl)[0][0], np.where(z < upp_bl)[0][-1]

        

    def _update_profiles_dict(self, solver, atmosphere, vel_adjust_factor, first=False):
        #Get solver states
        T1 = solver.state['T1']
        T1_z = solver.state['T1_z']
        P1   = solver.state['p1']


        z = atmosphere._new_field()
        z['g'] = atmosphere.z
        z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        z = z['g']

        bl_bot, bl_top, bl_bot_ind, bl_top_ind = self._find_BL_thick(z, atmosphere)

        bl_thick = (bl_bot + atmosphere.Lz - bl_top)/2

        # Update temperature fields for next solve. May need to add some logic here if nx == nz.
        init_kappa_flux = -atmosphere.P * self.profiles_dict['T_z_IVP']
        init_enth_flux = 1*self.profiles_dict['enth_flux_IVP']
        T1_z.set_scales(self.nz/atmosphere.nz, keep_data=True)
        self.profiles_dict['T_z_IVP'] += T1_z['g']

        # Adjust the enthalpy flux so that it on average carries all atmospheric flux in the bulk.
        kappa_flux          = -atmosphere.P * self.profiles_dict['T_z_IVP']
        tot_flux            = kappa_flux + self.profiles_dict['enth_flux_IVP']
        xs = z[(z>bl_bot)*(z<bl_top)]
        ys = self.profiles_dict['enth_flux_IVP'][(z>bl_bot)*(z<bl_top)]

        powers = np.arange(1)
        for i in range(len(powers)):
            n = 1+i
            p = np.polyfit(xs, ys, n)
            line = np.zeros_like(xs)
            for j in range(n+1):
                line += p[j]*xs**(n-j)
            powers[i] = np.sum(np.abs( (line - ys) / ys))
        pow_fit = np.argmin(powers) + 1
        p = np.polyfit(xs, ys, pow_fit)
        line = np.zeros_like(z)
        for i in range(pow_fit+1):
            line += p[i]*z**(pow_fit-i)
#        self.profiles_dict['enth_flux_IVP']  /= line
#        plt.plot(z, self.profiles_dict['enth_flux_IVP'], label='post_detrend')
#        self.profiles_dict['enth_flux_IVP'][:bl_bot_ind] /= np.max(self.profiles_dict['enth_flux_IVP'][:bl_bot_ind])
#        plt.plot(z, self.profiles_dict['enth_flux_IVP'], label='post_left')
#        self.profiles_dict['enth_flux_IVP'][bl_top_ind:] /= np.max(self.profiles_dict['enth_flux_IVP'][bl_top_ind:])
#        plt.plot(z, self.profiles_dict['enth_flux_IVP'], label='post_right')
#        self.profiles_dict['enth_flux_IVP'][bl_bot_ind:bl_top_ind] = 1
#        plt.plot(z, self.profiles_dict['enth_flux_IVP'], label='post_mid')
#        plt.legend(loc='best')
#        plt.savefig('{}/enth_flux_detrend_{}.png'.format(self.plot_dir, self.plot_count))
#        plt.close()
        sigma = bl_thick
        self.profiles_dict['enth_flux_IVP'] = (1 - np.exp(-z**2 / 2 / (sigma)**2) - np.exp(-(z-atmosphere.Lz)**2 / 2 / (sigma)**2) )

#        self.profiles_dict['enth_flux_IVP'] = savgol_filter(self.profiles_dict['enth_flux_IVP'], 11, 3)
        self.profiles_dict['enth_flux_IVP'] *= tot_flux

#        self.profiles_dict['enth_flux_IVP'][self.profiles_dict['enth_flux_IVP'] > tot_flux] = tot_flux[self.profiles_dict['enth_flux_IVP'] > tot_flux]

        if not isinstance(self.plot_dir, type(None)):
            plt.plot(z, init_kappa_flux + init_enth_flux)
            plt.plot(z, init_kappa_flux)
            plt.plot(z, init_enth_flux)
            plt.plot(z, self.profiles_dict['enth_flux_IVP'])
            plt.axvline(bl_bot)
            plt.axvline(bl_top)
            plt.savefig('{}/fluxes_{:04d}.png'.format(self.plot_dir, self.plot_count))
            plt.close()
            for fd in self.FIELDS.keys():
                if self.FIELDS[fd][1] == 0:
                    plt.plot(z, self.profiles_dict[fd])
                    plt.savefig('{}/{}_{:04d}.png'.format(self.plot_dir, fd, self.plot_count))
                    plt.close()

        self.plot_count += 1



        
        #In the simulation, adjust the velocity field by a constant value according to the mean drop in enth flux.
#        print(self.profiles_dict['enth_flux_IVP'] / init_enth_flux)
        vel_adjust = np.median(self.profiles_dict['enth_flux_IVP'] / init_enth_flux)
        vel_adjust_factor *= vel_adjust

        #Report
        avg_change = np.mean(T1['g'])
        logger.info('avg change T1: {}'.format(avg_change))
        logger.info('vel_adj factor: {}'.format(vel_adjust))

        return vel_adjust_factor, np.mean(np.abs(T1['g']))



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
            avg_change = 1e10
            vel_adjust_factor = 1
            first=True
            while avg_change > 1e-7:


                atmosphere = self.atmosphere_class(dimensions=1, comm=MPI.COMM_SELF, **atmosphere_kwargs)
                atmosphere.problem = de.NLBVP(atmosphere.domain, variables=['T1', 'T1_z','p1'], ncc_cutoff=tolerance)

                #Zero out old varables to make atmospheric substitutions happy.
                old_vars = ['u', 'w', 'dx(A)', 'uz', 'wz']
                for sub in old_vars:
                    atmosphere.problem.substitutions[sub] = '0'

                atmosphere._set_parameters(*diffusivity_args)
                atmosphere._set_subs()
 
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

                vel_adjust_factor, avg_change =\
                        self._update_profiles_dict(solver, atmosphere, vel_adjust_factor, first=first)

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
                
                if first:
                    return_dict['T1_IVP'] += self.profiles_dict['T1_IVP']
                    return_dict['T1_z_IVP'] += self.profiles_dict['T1_z_IVP']
                    return_dict['p_IVP'] += self.profiles_dict['p_IVP']
     
                first=False
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


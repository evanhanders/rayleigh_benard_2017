#/usr/bin/env python
import os
import sys
import time
import warnings
import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import dedalus.public as de
import h5py
from mpi4py import MPI
from scipy import optimize as opt
from scipy import interpolate


from tools.eigentools.eigentools import Eigenproblem, CriticalFinder
from boussinesq_dynamics import equations
BoussinesqEquations2D = equations.BoussinesqEquations2D


CW = MPI.COMM_WORLD
warnings.filterwarnings("ignore")

class OnsetSolver:
    """
    This class finds the onset of convection in a specified atmosphere
    (currently multitropes and polytropes) using a specified equation
    set (currently FC Navier Stokes is implemented).

    NOTE: This class currently depends on Evan Anders' branch of eigentools,
    found at https://bitbucket.org/evanhanders/eigentools
    """

    def __init__(self, ra_steps=(1, 1e3, 40, True),
                 kx_steps=(0.01, 1, 40, True), ky_steps=None, threeD=False, atmo_kwargs={}, eqn_args=[],
                 eqn_kwargs={}, bc_kwargs={}):
        """
        Initializes the onset solver by specifying the equation set to be used
        and the type of atmosphere that will be solved on.  Also specifies
        The range of Rayleigh numbers and horizontal wavenumbers to examine.

        Keyword Arguments:
            ra_steps    - A tuple containing four elements:
                            1. Min Ra to solve eigenproblem at
                            2. Max Ra to solve eigenproblem at
                            3. Num steps in Ra space between min/max
                            4. A bool.  If True, step through Ra in log space,
                               if False, step through in linear space.
            kx_steps    - A tuple containing four elements.  All elements are
                          the same as in ra_steps, just for horizontal
                          wavenumber.  Note here that kx=1 means that the
                          horizontal wavemode being examined is the wavemode
                          corresponding to a wavelength of the domain depth.
            atmo_kwargs - A Python dictionary, containing all default
                          information for the atmosphere, including 
                          number of z points, superadiabaticity, etc.
            eqn_args     - A list of arguments to be passed to set_equations
            eqn_kwargs   - A dictionary of keyword arguments to be passed to 
                           set_equations
            bc_kwargs    - A list of keyword arguments to be passed to 
                           set_BC
        """
        self._ra_steps   = ra_steps
        self._kx_steps   = kx_steps
        self._ky_steps   = ky_steps
        self.threeD      = threeD

        self._atmo_kwargs = atmo_kwargs
        self._eqn_args    = eqn_args
        self._eqn_kwargs  = eqn_kwargs
        self._bc_kwargs   = bc_kwargs
        self.cf = CriticalFinder(self.solve_problem, CW)

    def find_crits(self, tol=1e-3, pts_per_curve=1000, 
                   out_dir='./', out_file=None, load=False, exact=False):
        """
        Steps through all tasks and solves eigenvalue problems for
        the specified parameters.  If no tasks are specified, only
        the default parameters are solved for over the given kx/ra
        range.

        Keyword Arguments:
            tol             - The convergence tolerance for the iterative crit
                              finder (e.g., how little can the answer change
                              between steps before we're happy)
            pts_per_curve   - # of points to use on interpolated critical curve
            out_dir         - Output directory of information files
            out_file        - Name of information file.  If None, auto generate.
        """
        self._data = dict()

        if self.cf.rank == 0 and not os.path.exists('{:s}'.format(out_dir)):
            os.mkdir('{:s}'.format(out_dir))
        if out_file == None:
            out_file = '{:s}/hydro_onset'.format(out_dir)

        self.cf = CriticalFinder(self.solve_problem, CW)
        # If no tasks specified, set the atmospheric defaults,
        # find the crits, and store the curves
        self.atmo_kwargs = self._atmo_kwargs
        mins, maxs, ns, logs = [],[],[],[]
        for l in (self._ra_steps, self._kx_steps, self._ky_steps):
            if type(l) == type(None):
                continue
            mins.append(l[0])
            maxs.append(l[1])
            ns.append(l[2])
            logs.append(l[3])
        mins, maxs = np.array(mins), np.array(maxs)
        ns, logs   = np.array(ns, dtype=np.int64), np.array(logs)
        if load:
            try:
                self.cf.load_grid('{:s}/{:s}.h5'.format(out_dir, out_file), logs=logs)
            except:
                self.cf.grid_generator(mins, maxs, ns, logs=logs)
                if self.cf.comm.rank == 0:
                    self.cf.save_grid('{:s}/{:s}'.format(out_dir, out_file))
        else:
            self.cf.grid_generator(mins, maxs, ns, logs=logs)
            if self.cf.comm.rank == 0:
                self.cf.save_grid('{:s}/{:s}'.format(out_dir, out_file))
        self.cf.root_finder()
        if self.cf.comm.rank == 0:
            self.cf.plot_crit(title= '{:s}/{:s}'.format(out_dir, out_file), xlabel='kx', ylabel='Ra', transpose=True)

        if exact:
            crits = self.cf.exact_crit_finder()
        else:
            crits = self.cf.crit_finder()

        if len(crits) == 2:
            ra_crit, kx_crit = crits
            if self.cf.rank == 0:
                logger.info('Critical value found at ra: {:.5g}, kx: {:.5g}'.format(ra_crit, kx_crit)) 
        elif len(crits) == 3:
            ra_crit, kx_crit, ky_crit = crits
            k_tot = np.sqrt(kx_crit**2 + ky_crit**2)
            if self.cf.rank == 0:
                logger.info('Critical value found at ra: {:.5g}, kx: {:.5g}, ky: {:.5g}, ktot = {:.5g}'.format(ra_crit, kx_crit, ky_crit, k_tot)) 
        if self.cf.comm.rank == 0:
            self.cf.save_grid('{:s}/{:s}'.format(out_dir, out_file))
            self.cf.plot_crit(title= '{:s}/{:s}'.format(out_dir, out_file), xlabel='kx', ylabel='Ra', transpose=True)
       
    def solve_problem(self, ra, kx, ky=0):
        """
        Given a horizontal wavenumber and Rayleigh number, create the specified
        atmosphere, solve an eigenvalue problem, and return information about 
        the growth rate of the solution.

        Arguments:
            kx  - The horizontal wavenumber, in units of 2*pi/Lz, where Lz is the
                  depth of the atmosphere.
            ra  - The Rayleigh number to be used in solving the atmosphere.
        """

        self.atmosphere = BoussinesqEquations2D(dimensions=1, comm=MPI.COMM_SELF,
                                   grid_dtype=np.complex128, **self.atmo_kwargs)
        kx_real = kx*2*np.pi/self.atmosphere.Lz

        #Set the eigenvalue problem using the atmosphere
        self.atmosphere.set_EVP(ra, 
                *self._eqn_args, kx=kx_real, **self._eqn_kwargs)
        self.atmosphere.set_BC(**self._bc_kwargs)
        problem = self.atmosphere.get_problem()

        #Solve using eigentools Eigenproblem
        self.eigprob = Eigenproblem(problem)
        max_val, gr_ind, freq = self.eigprob.growth_rate({})
        #Initialize atmosphere
        if self.cf.rank == 0:
            logger.info('Solving for onset with ra {:.8g} / kx {:.8g} / ky {:.8g} on proc 0'.\
                    format(ra, kx, ky))
            logger.info('Maximum eigenvalue found at those values: {:.8g}'.format(max_val))
        

        if not np.isnan(max_val):
            val = max_val + 1j*freq
            if type(val) == np.ndarray:
                return val[0]
            else:
                return val
        else:
            return np.nan

if __name__ == '__main__':
    help(OnsetSolver)

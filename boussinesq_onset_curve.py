"""
Dedalus script for finding the onset of compressible convection in a 
polytropic or multitropic atmosphere

Usage:
    FC_onset_curve.py [options] 

Options:
    --rayleigh_start=<Rayleigh>         Rayleigh number start [default: 1e2]
    --rayleigh_stop=<Rayleigh>          Rayleigh number stop [default: 1e4]
    --rayleigh_steps=<steps>            Integer number of steps between start 
                                         and stop Ra   [default: 5]
    --kx_start=<kx_start>               kx to start at [default: 0.1]
    --kx_stop=<kx_stop>                 kx to stop at  [default: 1]
    --kx_steps=<kx_steps>               Num steps in kx space [default: 5]

    --ky_start=<ky_start>               ky to start at [default: 0.1]
    --ky_stop=<ky_stop>                 ky to stop at  [default: 1]
    --ky_steps=<ky_steps>               Num steps in kx space [default: 20]

    --nz=<nz>                           z (chebyshev) resolution [default: 32]

    --bcs=<bcs>                         Boundary conditions ('fixed', 'mixed', 
                                            or 'flux') [default: fixed]
    --vel_bcs=<bcs>                     Velocity Boundary conditions ('stress_free', 'no_slip') [default: no_slip] 
    --Prandtl=<Pr>                      Prandtl number [default: 1]


    --3D                                If flagged, use 3D eqns and search kx & ky
    --2.5D                              If flagged, use 3D eqns with ky = 0

    --load                              If flagged, attempt to load data from output file
    --exact                             If flagged, after doing the course search + interpolation,
                                            iteratively solve for the exact critical using
                                            optimization routines
    --out_dir=<out_dir>                 Base output dir [default: ./]
"""
import logging
logger = logging.getLogger(__name__)
from docopt import docopt
from onset_solver import OnsetSolver
import numpy as np

args = docopt(__doc__)

logger.info("Solving rayleigh benard onset")
file_name = 'boussinesq_onset'



##########################################
#Set up ra / kx / ky spans
ra_log, kx_log, ky_log = False, False, False

ra_start = float(args['--rayleigh_start'])
ra_stop = float(args['--rayleigh_stop'])
ra_steps = float(args['--rayleigh_steps'])
if np.abs(ra_stop/ra_start) > 10:
    ra_log = True

kx_start = float(args['--kx_start'])
kx_stop = float(args['--kx_stop'])
kx_steps = float(args['--kx_steps'])
if np.abs(kx_stop/kx_start) > 10:
    kx_log = True

ky_start = float(args['--ky_start'])
ky_stop = float(args['--ky_stop'])
ky_steps = float(args['--ky_steps'])
if np.abs(ky_stop/ky_start) > 10:
    ky_log = True

############################################
#Set up BCs
bcs = args['--bcs']
vel_bcs = args['--vel_bcs']
fixed_flux = False
fixed_T    = False
mixed_T    = False
if bcs == 'mixed':
    mixed_T = True
    file_name += '_mixedBC'
elif bcs == 'fixed':
    fixed_T = True
    file_name += '_fixedTBC'
else:
    fixed_flux = True
    file_name += '_fluxBC'

no_slip = False
stress_free = False
if vel_bcs == 'stress_free':
    stress_free = True
    file_name += '_stress_free'
else:
    no_slip = True
    file_name += '_no_slip'

bc_kwargs = {'fixed_temperature': fixed_T,
               'mixed_flux_temperature': mixed_T,
               'fixed_flux': fixed_flux,
               'stress_free': stress_free,
               'no_slip':   no_slip}



##########################################
#Set up defaults for the atmosphere
nz = args['--nz'].split(',')
nz = [int(n) for n in nz]
atmo_kwargs = { 'stream_function':             stress_free,
                    'nz':                   nz,
                    'Lz':                   1.}


##############################################
#Setup default arguments for equation building
eqn_kwargs = dict()
eqn_args = [float(args['--Prandtl'])]

#####################################################
#Initialize onset solver
if args['--3D']:
    file_name += '_3D'
    ky_steps = (ky_start, ky_stop, ky_steps, ky_log)
    threeD = True
elif args['--2.5D']:
    file_name += '_2.5D'
    ky_steps = None
    threeD = True
else:
    file_name += '_2D'
    ky_steps = None
    threeD = False

solver = OnsetSolver(
            threeD=threeD,
            ra_steps=(ra_start, ra_stop, ra_steps, ra_log),
            kx_steps=(kx_start, kx_stop, kx_steps, kx_log),
            ky_steps=ky_steps,
            atmo_kwargs=atmo_kwargs,
            eqn_args=eqn_args,
            eqn_kwargs=eqn_kwargs,
            bc_kwargs=bc_kwargs)

#############################################
#Crit find!
out_dir = args['--out_dir']
load    = args['--load']
exact   = args['--exact']
solver.find_crits(out_dir=out_dir, out_file='{:s}'.format(file_name), load=load, exact=exact)

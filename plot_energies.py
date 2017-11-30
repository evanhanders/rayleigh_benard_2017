"""
Plot energies from joint analysis files.

Usage:
    plot_energies.py join <base_path>
    plot_energies.py <files>... [--output=<output>]

Options:
    --output=<output>  Output directory; if blank a guess based on likely case name will be made

"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__.split('.')[-1])

import analysis

def plot_energies(data, times, output_path='./'):
    t = times
        
    figs = {}
    
    fig_energies = plt.figure(figsize=(16,8))
    ax1 = fig_energies.add_subplot(2,1,1)
    plt.grid()
    ax1.semilogy(t, data['KE'], label="KE")
    ax1.semilogy(t, data['IE'], label="IE")
    ax1.semilogy(t, data['TE'], label="TE")
    ax1.legend()
    
    ax2 = fig_energies.add_subplot(2,1,2)
    plt.grid()
    ax2.plot(t, data['KE'], label="KE")
    ax2.plot(t, data['IE'], label="IE")
    ax2.plot(t, data['TE'], label="TE")
    ax2.legend()
    figs["energies"]=fig_energies

    fig_KE = plt.figure(figsize=(16,8))
    ax1 = fig_KE.add_subplot(1,1,1)
    plt.grid()
    ax1.plot(t, data['KE'], label="KE")
    ax1.plot(t, data['IE']-data['IE'][0], label="IE-IE$_0$")
    ax1.plot(t, data['TE']-data['TE'][0], label="TE-TE$_0$", color='black')
    ax1.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    figs["fluctuating_energies"] = fig_KE

    fig_KE_2 = plt.figure(figsize=(16,8))
    ax1 = fig_KE_2.add_subplot(1,1,1)
    plt.grid()
    ax1.plot(t, data['KE'], label="KE total")
    ax1.plot(t, data['KE_fluc'], label="KE fluc")
    ax1.plot(t, data['KE'] - data['KE_fluc'], label="KE mean")
    ax1.plot(t, data['KE_x_fluc'], label=r"KE$_x$ fluc", linestyle='dashed')
    ax1.plot(t, data['KE_z_fluc'], label=r"KE$_z$ fluc", linestyle='dashed')    
    ax1.plot(t, data['IE']-data['IE'][0], label="IE-IE$_0$")
    ax1.plot(t, data['TE']-data['TE'][0], label="TE-TE$_0$", color='black')
    ax1.legend()
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    figs["fluctuating_energies_breakdown"] = fig_KE_2

    fig_KE_only = plt.figure(figsize=(16,8))
    ax1 = fig_KE_only.add_subplot(2,1,1)
    plt.grid()
    ax1.plot(t, data['KE'], label="KE")
    ax1.legend()
    ax1.set_ylabel("energy")
    ax2 = fig_KE_only.add_subplot(2,1,2)
    plt.grid()
    ax2.semilogy(t, data['KE'], label="KE")
    ax2.legend()
    ax2.set_xlabel("time")
    ax2.set_ylabel("energy")
    figs["KE"] = fig_KE_only
 
    fig_log = plt.figure(figsize=(16,8))
    ax1 = fig_log.add_subplot(1,1,1)
    plt.grid()
    ax1.semilogy(t, data['KE'], label="KE")
    ax1.semilogy(t, np.abs(data['IE']-data['IE'][0]), label="|IE-IE$_0$|")
    ax1.semilogy(t, np.abs(data['TE']-data['TE'][0]), label="|TE-TE$_0$|", color='black')
    ax1.set_xlabel("time")
    ax1.set_ylabel("energy")
    ax1.legend()
    figs["log_fluctuating_energies"] = fig_log

    fig_Nu = plt.figure(figsize=(16,8))
    ax1 = fig_Nu.add_subplot(1,1,1)
    plt.grid()
    ax1.semilogy(t, data['Nu'], label="Nu")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Nu")
    ax1.legend()
    figs["nusselt_fixedT"] = fig_Nu

    for key in figs.keys():
        figs[key].savefig(output_path+'scalar_{}.png'.format(key))
    
def main(files, output_path='./'):
    data = analysis.Scalar(files)
    plot_energies(data.data, data.times, output_path=output_path)
    

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    if args['join']:
        post.merge_analysis(args['<base_path>'])
    else:
        if args['--output'] is not None:
            output_path = pathlib.Path(args['--output']).absolute()
        else:
            data_dir = args['<files>'][0].split('/')[0]
            data_dir += '/'
            output_path = pathlib.Path(data_dir).absolute()
        # Create output directory if needed
        with Sync() as sync:
            if sync.comm.rank == 0:
                if not output_path.exists():
                    output_path.mkdir()
        logger.info("output to {}".format(output_path))
        main(args['<files>'], output_path=str(output_path)+'/')



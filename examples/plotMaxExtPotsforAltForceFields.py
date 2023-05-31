# %%
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import glob

import matplotlib.gridspec as gridspec
import itertools
# from density_set import DensitySet

def main() -> None:

    particle = 1 # on a 0 index basis
    N_bulk = 0.015


    # Parameter initialization update these if expanding
    fig_name = 'MaxPotComparison_'  # include traceability here to training version
    # fig_title = 'Simulation RDFs of NNP Potentials '
    labels = ['FT', 'Standard NNP','Ext Pot NNP']  # legend labels
    directs = [r'/home/kamron/mdext_KF/examples/molten_salt/data6Nima40angBMH/',
                r'/home/kamron/mdext_KF/examples/molten_salt/data6Nima40angDPMDreg/',
                r'/home/kamron/mdext_KF/examples/molten_salt/data6Nima40angDPMDpert/']
                
    figLabel = ['a', 'b']
    # titles = []  # titles of filename/MD scenario
    # pattern = r'*dat'
    # os.getcwd()

    # Identify all dat files and read them in as data1 and data2
    # First dir sets all the titles up so be sure to have the same dat files in the second dir

    # Get titles
    os.chdir(r'/home/kamron/mdext_KF/examples/molten_salt/') 

    def GetData(direct, extPot):

        with h5py.File(direct+f"test-U{extPot:+.1f}.h", "r") as fp:
            r = np.array(fp["r"])
            n = np.array(fp["n"])
            # V = np.array(fp["V"])

        return r, n  # list containing 2D layers of each scenario

    
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.13)

    for ax_ind, ax in enumerate(axs.flat):  # order somehow manages to be lq hp crystal so reorder it
        #    print(ax)
        #     ax.subplot(2,3,i+1, sharex=ax1, sharey=ax1)
        # Get data

        for i, direct in enumerate(directs):
            plt.sca(ax)
            if ax_ind == 0:
                #Bottom - repulsive
                r,n = GetData(direct, 5.0)  # , potential eV
                plt.ylim((0,2))
                
            else:
                #Top - attractive
                r,n = GetData(direct, -5.0)
                plt.ylim((0,4))
                plt.xlabel("z [$\AA$]",fontsize=14)    

            plt.plot(r, n[:,particle]/N_bulk, label=labels[i])
            plt.text(-0.2, 1.01, f"({figLabel[ax_ind]})", ha="left", va="top",
                transform=ax.transAxes, fontsize="large", fontweight="bold")
            
            plt.ylabel("$n_{Na}(z)/n_{bulk}$",fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

        # ax.set_title(titles[j])
        ax.grid(True)
        # j-=1
    
    axs[0].legend()



    fig.savefig(fig_name+'combo.pdf', bbox_inches='tight')
    
    sys.exit(0)

    # Repulsive
    # plt.figure(figsize=(8,6), dpi=300)
#----------------
    for i, direct in enumerate(directs):
        r,n = GetData(direct, 5.0)
        plt.plot(r, n[:,particle]/N_bulk, label=labels[i])
        plt.ylim((0,4))
        plt.legend()
        plt.xlabel("z [$\AA$]",fontsize=11,fontname="Times New Roman")
        plt.ylabel("$n_{Na}(z)/n_{bulk}$",fontsize=11,fontname="Times New Roman")

    plt.savefig(fig_name+'attract.pdf', bbox_inches='tight')



    plt.figure(figsize=(8,6), dpi=300)

    for i, direct in enumerate(directs):
        r,n = GetData(direct, -5.0)
        plt.plot(r, n[:,particle]/N_bulk, label=labels[i])
        plt.ylim((0,4))
        plt.legend()
        plt.xlabel("z [$\AA$]",fontsize=11,fontname="Times New Roman")
        plt.ylabel("$n_{Na}(z)/n_{bulk}$",fontsize=11,fontname="Times New Roman")

    plt.savefig(fig_name+'repulse.pdf', bbox_inches='tight', dpi=300)


    plt.figure(figsize=(8,6), dpi=300)




if __name__ == "__main__":
    main()

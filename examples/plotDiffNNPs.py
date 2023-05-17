# %%
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import glob

endRange = 5.0
stepSize = 0.5

particle = 1 # on a 0 index basis
N_bulk = 0.015



    plt.plot(r,n[:,particle]/(N_bulk),color=cmap(normalize(Ui)), lw=1)
    plt.xlabel("z [$\AA$]",fontsize=11,fontname="Times New Roman")
    plt.ylabel("$n_{Na}(z)/n_{bulk}$",fontsize=11,fontname="Times New Roman")
    # plt.ylabel("$n_{H}(z)/n_{bulk}$",fontsize=11,fontname="Times New Roman")
    
    # plt.legend()
    plt.legend( prop={
            'family': 'Times New Roman', "size": 7, 'stretch': 'normal'})    # plt.title('Classical, -0.4 eV',fontsize=12,fontname="Times New Roman")
    plt.xticks(fontsize=11,fontname="Times New Roman")
    plt.yticks(fontsize=11,fontname="Times New Roman")

# plt.plot(r, V, color="k", lw=1, ls="dashed")  # plot largest shaped V
# plt.xlim([r.min(),r.max()])
plt.ylim([0,5])

# --- add colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=normalize)
sm.set_array([])
plt.colorbar(sm, label=r"Perturbation strength, $\lambda$")
figure = plt.gcf()
# figure.set_size_inches(3.35, 2.2)
plt.savefig('plotMany.pdf', dpi=600, bbox_inches='tight')

# %%
# %%
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import glob
# import shutil

# Parameter initialization update these if expanding
fig_name = 'PBE_9b PBEsol_2b PBED2_1 PBED3_1 FT comparison.png'  # include traceability here to training version
# fig_title = 'Simulation RDFs of NNP Potentials '
labels = ['PBE', 'PBESol','PBE-D2','PBE-D3','FT']  # legend labels
directs = [r'/global/homes/k/kamron/mdext_KF/examples/molten_salt/D2ClNaPerturbTrain7r11/',
            r'/global/homes/k/kamron/mdext_KF/examples/molten_salt/PBED2NaClTrain1AllTraining',
            
a = 'r'  # column names for reading in the csv file
b = 'g12'
titles = []  # titles of filename/MD scenario
pattern = r'*dat'
# os.getcwd()

# Identify all dat files and read them in as data1 and data2
# First dir sets all the titles up so be sure to have the same dat files in the second dir

# Get titles
os.chdir(r'/home/kamron/NaCl/integrate_data/nnpPBE') 
for pathAndFilename in glob.iglob(os.path.join(os.getcwd(), pattern)):
    Filename = os.path.basename(pathAndFilename)  # typically crystal1.rdf.dat
    title, ext = os.path.splitext(Filename)
    titles.append(title[:-4])  # trim middle ext off - could just do this to begin with
    

def GetData(direct):
# Get data function getting each scenario and data for RDFs
    os.chdir(direct)
    data = []
    pattern = r'*dat'
    for pathAndFilename in glob.iglob(os.path.join(os.getcwd(), pattern)):
        Filename = os.path.basename(pathAndFilename)
        # read csv and plot Test MSE
        
    with h5py.File(f"test-U{Ui:+.1f}.h", "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])

        data.append(pd.read_csv(Filename, skiprows=1, usecols=[0,2], delimiter=' ', names=[a,b]))
    return data  # list containing 2D layers of each scenario


# %%
# PLOT
# fname = filenm + '.png'
# plt.figure(figsize=(8,6), dpi=300)
# fig, axs = plt.subplots(2, 3, figsize=(10, 7), dpi=300, sharex=True, sharey=True)
fig, axs = plt.subplots(3, 1, figsize=(8, 9), dpi=300, sharex=True, sharey=True)
# fig.suptitle(fig_title)

# ax1 = fig.add_subplot(2,3,1)
# ax1 = plt.subplot(2,3,1)
# ax1.plot(dataPBE[i]['r'], dataPBE[i]['g12'], label = 'PBE')
# ax1.plot(dataPBESol[i]['r'], dataPBESol[i]['g12'], label = 'PBESol')
# ax1.set_xlabel('r [A]')
# ax1.set_ylabel('g(r)')

j = 2
for i, ax in enumerate(axs.flat):  # order somehow manages to be lq hp crystal so reorder it
    #    print(ax)
    #     ax.subplot(2,3,i+1, sharex=ax1, sharey=ax1)
    # Get data
    for p, path in enumerate(directs):
        data = GetData(path)
        ax.plot(data[j]['r'], data[j]['g12'], label=labels[p])

    ax.set_title(titles[j])
    ax.grid(True)
    j-=1
axs[0].legend()
axs[1].set_ylabel('g(r)')
axs[2].set_xlabel('r [Ang]')
os.chdir(r'/home/kamron/NaCl/integrate_data')
fig.savefig(fig_name)
fig.show()
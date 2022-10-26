import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
from glob import glob

if True:
    import os
    os.chdir(r'/home/kamron/ToSync/AIMP/mdext')
    # filename = "NaCl_U-0.1.h5"
    n_bulk = 0.015


if False:
    if len(sys.argv) < 2:
        print("Usage: python plot.py <file1.h5> [<n_bulk>]")
        exit(1)

    filename = sys.argv[1]
    n_bulk = float(sys.argv[2]) if (len(sys.argv) > 2) else None

# f = h5py.File(filename, "r")
# f.keys()


for filename in sorted(glob('*.h5')):
    with h5py.File(filename, "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
    axes[0].plot(r, n[:,0], rasterized=True, label='Na')
    axes[0].plot(r, n[:,1], rasterized=True, label='Cl')
    axes[0].legend()
    if n_bulk is not None:
        axes[0].axhline(n_bulk, color='k', ls='dotted')
    axes[0].set_ylabel("Density $(1/Ang^3)$")
    axes[1].plot(r, V[:,0], rasterized=True, label='Na')
    axes[1].plot(r, V[:,1], rasterized=True, label='Cl')
    axes[1].legend()
    axes[1].set_ylabel("Potential (eV)")
    axes[1].set_xlabel("r")
    plt.savefig(filename[:-3]+".svg")
    # plt.show()

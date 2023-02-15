import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
from glob import glob

def trapz(f: np.ndarray, h: float) -> np.ndarray:
    """Cumulative trapezoidal integral of a function sampled at spacing `h`."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (f[:-1] + f[1:])) * h))


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

nAll = []
VAll = []

for filename in sorted(glob('*.h5')):
    with h5py.File(filename, "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])
        # np.shape(V) # n and V are 200,2  Na and Cl are 2nd dimension
    if False:
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
        plt.savefig(filename[:-3]+".pdf", bbox_inches='tight')
        # plt.show()

    # Compile
    nAll.append(n[:,0])  # appending Na only
    # VAll.append(V[:,0])

nAll = np.stack(nAll)

lbda = np.array([-.5,-.4,-.3,-.2,-.1,.1,.2,.3,.4,.5])  # don't know why 0 was missing...
Vselect = V[:,0]/lbda[-1]  # V/U0


# --- thermodynamic integration
dr = r[1] - r[0]
integrand = (nAll @ Vselect.T) * dr  # 1/A*eV*A
dlbda = 0.1 # change in U0
E_TI = trapz(integrand, dlbda)
E_TI -= np.interp(0.0, lbda, E_TI)  # difference from bulk
plt.plot(lbda, E_TI, "r+", label="TI")
plt.axhline(0, color="k", lw=1, ls="dotted")
plt.axvline(0, color="k", lw=1, ls="dotted")
plt.legend()
plt.xlim(lbda.min(), lbda.max())
plt.xlabel(r"Perturbation strength, $V_0$")
plt.ylabel(r"Free energy change, $\Delta\Phi$")
plt.savefig("NaCl_TI.pdf", bbox_inches='tight')
plt.show()

np.shape(Vselect) # 200
np.shape(integrand)
np.shape(E_TI)
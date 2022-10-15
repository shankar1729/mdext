import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python plot.py <file1.h5> [<n_bulk>]")
    exit(1)

filename = sys.argv[1]
n_bulk = float(sys.argv[2]) if (len(sys.argv) > 2) else None

with h5py.File("test.h5", "r") as fp:
    r = np.array(fp["r"])
    n = np.array(fp["n"])
    V = np.array(fp["V"])

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
axes[0].plot(r, n)
if n_bulk is not None:
    axes[0].axhline(n_bulk, color='k', ls='dotted')
axes[0].set_ylabel("Density")
axes[1].plot(r, V)
axes[1].set_ylabel("Potential")
axes[1].set_xlabel("r")
plt.show()

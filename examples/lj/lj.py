from mdext import log
from mdext.md import MD
import numpy as np
import matplotlib.pyplot as plt

md = None  # global object used in lammps thermo callback (created in main)
extpot = None   # global object used in lammps force callback (created in main)


def main() -> None:

    # Current simulation parameters:
    T = 298.
    seed = 12345
    U0 = +5.   # Amplitude of the external potential
    sigma = 2. # Width of the external potential

    # Initialize and run simulation:
    global md
    md = MD(T=T, seed=seed, U0=U0, sigma=sigma)
    md.setup_thermo_callback()
    md.run(2, "equilibration")
    md.reset_stats()
    md.run(5, "collection")
    
    # Plot and compare to analytical result:
    if md.is_head:
        # normalize
        plt.plot(md.z, md.density / md.i_cycle)
        plt.axhline(0.03334, color='k', ls='dotted')
        plt.show()

    
if __name__ == "__main__":
    main()

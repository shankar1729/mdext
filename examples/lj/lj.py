import mdext
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:

    # Current simulation parameters:
    T = 298.
    seed = 12345
    U0 = +5.   # Amplitude of the external potential
    sigma = 2. # Width of the external potential

    # Initialize and run simulation:
    md = mdext.md.MD(T=T, seed=seed, U0=U0, sigma=sigma)
    md.run(2, "equilibration")
    md.reset_stats()
    md.run(5, "collection")
    
    # Plot density response:
    if md.is_head:
        plt.plot(md.z, md.density / md.i_cycle)
        plt.axhline(0.03334, color='k', ls='dotted')
        plt.show()

    
if __name__ == "__main__":
    main()

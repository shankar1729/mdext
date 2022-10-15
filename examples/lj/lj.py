"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
from lammps import PyLammps
from mdext import log


def main() -> None:

    # Current simulation parameters:
    T = 298.0  # K
    P = 1.0  # atm
    seed = 12345
    U0 = -5.   # Amplitude of the external potential (kcal/mol)
    sigma = 2. # Width of the external potential (A)

    # Initialize and run simulation:
    md = mdext.md.MD(
        setup=setup,
        T=T,
        P=P,
        seed=seed,
        potential=mdext.potential.Gaussian(U0, sigma),
        geometry_type=mdext.geometry.Spherical,
        n_atom_types=1,
        potential_type=1,
    )
    md.run(2, "equilibration")
    md.reset_stats()
    md.run(5, "collection", "test.h5")


def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct simulation box:
    L = np.array([30., 30., 30.])  # overall box dimensions
    lmp.region(
        f"sim_box block -{L[0]/2} {L[0]/2} -{L[1]/2} {L[1]/2} -{L[2]/2} {L[2]/2}"
        " units box"
    )
    lmp.create_box("1 sim_box")
    n_bulk = 0.03334  # bulk density of water at STP (in A^-3)
    n_atoms = int(np.round(n_bulk * L.prod()))
    lmp.create_atoms(f"1 random {n_atoms} {seed} sim_box")
    lmp.mass("1 18.")  # pretending this is a united-atom model of water

    # Interaction potential:
    lmp.pair_style("lj/cut 10")
    lmp.pair_coeff("1 1 1.0 2.97")  # Adjusted to get water bulk density
    lmp.pair_modify("tail yes")  # Critical for NPT to be stable

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")


if __name__ == "__main__":
    main()

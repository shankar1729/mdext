"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
from lammps import PyLammps
from mdext import log


def main() -> None:

    # Current simulation parameters (all in LJ units):
    T = 0.7  # in LJ epsilon
    P = 1.0  # in LJ epsilon/sigma^2
    seed = 12345
    U0 = -15.0  # Amplitude of the external potential (in LJ epsilon)
    sigma = 0.5  # Width of the external potential (in LJ sigma)

    # Initialize and run simulation:
    md = mdext.md.MD(
        setup=setup,
        T=T,
        P=P,
        seed=seed,
        potential=mdext.potential.Gaussian(U0, sigma),
        geometry_type=mdext.geometry.Planar,
        n_atom_types=1,
        potential_type=1,
        dimension=1,
        units="lj",
        timestep=0.01,
        Tdamp=0.5,
        Pdamp=1.0,
    )
    md.run(10, "equilibration")
    md.reset_stats()
    md.run(40, "collection", "test.h5")


def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct simulation box:
    Lz = 30.  # only box dimension that matters
    L = np.array([1., 1., Lz])  # overall box dimensions
    lmp.region(
        f"sim_box block -{L[0]/2} {L[0]/2} -{L[1]/2} {L[1]/2} -{Lz/2} {Lz/2}"
        " units box"
    )
    lmp.create_box("1 sim_box")
    n_bulk = 0.7   # in LJ 1/sigma
    n_atoms = int(np.round(n_bulk * Lz))
    lmp.region(f"atom_box block -0.0 0.0 -0.0 0.0 -{Lz/2} {Lz/2} units box")
    lmp.create_atoms(f"1 random {n_atoms} {seed} atom_box")
    lmp.mass("1 1.")

    # Interaction potential:
    lmp.pair_style("lj/cut 10")
    lmp.pair_coeff("1 1 1.0 1.0")
    # lmp.pair_modify("tail yes")

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")


if __name__ == "__main__":
    main()

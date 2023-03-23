"""Molten NaCl using SimpleNN trained to PBE+D3 at 1300 K."""
import mdext
import numpy as np
from lammps import PyLammps
from mdext import MPI, log


def main() -> None:

    # Current simulation parameters:
    T = 1300.0  # K
    P = 1.0  # bar
    seed = 12345
    U0 = 1.0  # Amplitude of the external potential (eV) + repulsive
    sigma = 1. # Width of the external potential (A)
    dpmdPot = f"deepmd {pot}.pb"

    # Initialize and run simulation:
    md = mdext.md.MD(
        setup=setup,
        T=T,
        P=None,
        seed=seed,
        potential=mdext.potential.Gaussian(U0, sigma),
        geometry_type=mdext.geometry.Planar,
        n_atom_types=2,
        potential_type=2,
        units="metal",
        timestep=0.002,
        Tdamp=0.1,
        Pdamp=0.1,
    )
    md.run(2, "equilibration")
    md.reset_stats()
    md.run(5, "collection", f"test-U{U0:+.1f}.h5")


def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct water box:
    L = [20.] * 3  # box dimensions
    file_liquid = "liquid.data"
    is_head = (MPI.COMM_WORLD.rank == 0)
    if is_head:
        # needs to be Cl1 Na2
        # seems like the bonds and angles need to be removed??
        mdext.make_liquid.make_liquid(
            pos_min=[-L[0]/2, -L[1]/2, -L[2]/2],
            pos_max=[+L[0]/2, +L[1]/2, +L[2]/2],
            out_file=file_liquid,
            N_bulk=0.015,
            masses=[35.45, 22.99],
            radii=[1.0, 1.0],
            atom_types=[1, 2],  
            atom_pos=[[0., 0., 1.3], [0., 0., -1.3]],
            bond_types=np.zeros((0,), dtype=int),
            bond_indices=np.zeros((0, 2), dtype=int),
            angle_types=np.zeros((0,), dtype=int),
            angle_indices=np.zeros((0, 3), dtype=int),
        )
    lmp.atom_style("full")
    lmp.read_data(file_liquid)

    # Interaction potential (SimpleNN):
    # lmp.pair_style("nn")
    # lmp.pair_coeff("* * potential_snn Na Cl")
    
    # plugin load libdeepmd_lmp.so
    # pair_style	deepmd DpmdPBED2_potential.pb
    # pair_coeff  * *	

    lmp.plugin("load libdeepmd_lmp.so")
    lmp.pair_style(dpmdPot)
    lmp.pair_coeff("* *")

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")
    # no dump if resetting stats for plotting
    # lmp.dump(f'write all custom 100 pylammps.dump id type x y z vx vy vz')


if __name__ == "__main__":
    main()

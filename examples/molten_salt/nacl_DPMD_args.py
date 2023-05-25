"""Molten NaCl using SimpleNN trained to PBE+D3 at 1300 K."""
import mdext
import numpy as np
from lammps import PyLammps
from mdext import MPI, log
import argparse


def main() -> None:

    # Get simulation parameters from the command line:
    parser = argparse.ArgumentParser(description="Molten NaCl in external potentials")
    parser.add_argument(
        '-U', '--U0', type=float, required=True,
        help="amplitude of Gaussian external potential in eV"
    )
    parser.add_argument(
        '-s', '--sigma', type=float, required=True,
        help="width of Gaussian external potential in Angstroms",
    )
    parser.add_argument(
        '-S', '--seed', type=int, default=12345,
        help="random number seed",
    )
    parser.add_argument(
        '-T', '--temperature', type=float, required=True,
        help="temperature in Kelvin",
    )
    parser.add_argument(
        '-P', '--pressure', type=float, required=False,
        help="pressure in bars",
    )
    parser.add_argument(
        '-o', '--output_file', type=str, required=True,
        help="HDF5 output filename",
    )
    parser.add_argument(
        '-p', '--potential_type', type=int, required=True,
        help="atom type to apply the potential to (1-based)"
    )
    geometry_map = {
        'spherical': mdext.geometry.Spherical,
        'cylindrical': mdext.geometry.Cylindrical,
        'planar': mdext.geometry.Planar,
    }
    parser.add_argument(
        '-g', '--geometry', choices=geometry_map.keys(), required=True,
        help="1D symmetry of external potential",
    )
    parser.add_argument(
        '--potential_filename', type=str, required=True,
        help="deepmd potential file name"
    )
    args = parser.parse_args()
    
    global dpmdPotLine
    dpmdPotLine=f'deepmd ../{args.potential_filename}'
    
    # Initialize and run simulation:
    md = mdext.md.MD(
        setup=setup,
        T=args.temperature,
        P=None,
        # P=args.pressure,
        seed=args.seed,
        potential=mdext.potential.Gaussian(args.U0, args.sigma),
        geometry_type=geometry_map[args.geometry],
        n_atom_types=2,
        potential_type=args.potential_type,
        units="metal",
        timestep=0.002,
        Tdamp=0.1,
        Pdamp=0.1,
    )
    # md.run(1, "equilibration")
    md.run(2, "equilibration")
    md.reset_stats()
    md.run(5, "collection", args.output_file)


def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct water box:
    L = [40.] * 3  # box dimensions
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
    lmp.pair_style(dpmdPotLine)
    lmp.pair_coeff("* *")

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")
    # no dump if resetting stats for plotting
    # lmp.dump(f'write all custom 100 pylammps.dump id type x y z vx vy vz')


if __name__ == "__main__":
    main()

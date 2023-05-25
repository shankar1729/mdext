
"""Molten NaCl using Fumi-Tosi potential at 1300 K."""
# import os
# os.chdir('/home/kamron/mdext_KFtesting/')

# import sys
# sys.path.insert(0, '/home/kamron/mdext_KF/src/')  # to allow testing local python packages 

import mdext
import numpy as np
from lammps import PyLammps
from mdext import MPI, log
from dataclasses import dataclass

def main() -> None:


    
    # Current simulation parameters:
    T = 1300.0  # K
    P = 1.0  # bar
    seed = 12345
    U0 = 1.0  # Amplitude of the external potential (eV)
    # U0 = +0.20  # Amplitude of the external potential (eV)
    sigma = 1. # Width of the external potential (A)
    setup = Setup(startfile)
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
    
    # md.run(2, "equilibration") # 10 for final
    # md.reset_stats()
    # md.run(5, "collection", f"test-U{U0:+.2f}.h5") # 100 for final
    md.lmp.write_data("test.data")

@dataclass
class Setup:
    startfile: str
    R: float = 0.0
    
    # replaces 
    # def __init__(self, startfile):
    #     self.startfile = startfile
        
    def __call__(self, lmp: PyLammps, seed: int) -> int:
        """Setup initial atomic configuration and interaction potential."""

        # Construct water box:
        L = [20.] * 3  # box dimensions
        file_liquid = self.startfile # "liquid.data"
        
        is_head = (MPI.COMM_WORLD.rank == 0)
        if is_head:
            # Cl1 Na2
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

        # Interaction potential (Fumi-Tosi w/ Ewald summation):
        lmp.set(f"type 1 charge -1")
        lmp.set(f"type 2 charge +1")
        lmp.pair_style("born/coul/long 9.0")
        lmp.pair_coeff("1 1 0.158223509 0.317 3.170 72.40215779 -145.4284714")  # Cl-Cl updated
        lmp.pair_coeff("2 2 0.263705848 0.317 2.340 1.048583006 -0.49932529") # Na-Na
        lmp.pair_coeff("1 2 0.210964679 0.317 2.755 6.99055303  -8.675775756")  # Na-Cl
        # pair_coeff 1 1 0.158223509 0.317 3.170 72.40215779 -145.4284714
        # pair_coeff 1 2 0.210964679 0.317 2.755 6.99055303  -8.675775756
        # pair_coeff 2 2 0.263705848 0.317 2.340 1.048583006 -0.49932529
        lmp.kspace_style("ewald 1e-5")



        # Initial minimize:
        log.info("Minimizing initial structure")
        lmp.minimize("1E-4 1E-6 10000 100000")
        # no dump if resetting stats for plotting
        # lmp.dump("write all custom 100 pylammps.dump id type x y z vx vy vz")

if __name__ == "__main__":
    main()

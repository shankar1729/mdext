import numpy as np
from mdext import MPI


class Planar:
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up planar Gaussian potential with peak `U0` and width `sigma`."""
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)
    
    def __call__(self, lps, ntimestep, nlocal, tag, x, f) -> None:
        '''Callback function to add external potential and collect data.'''
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        
        # Wrap positions periodically:
        pos = x/L  # to fractional coordinates
        pos -= np.floor(0.5 + pos)  # wrap to [-0.5, 0.5)
        pos *= L  # back to Cartesian coordinates

        # Add gaussian potential:
        z = pos[:, 2]
        # --- energy:
        E = self.U0 * np.exp(-0.5 * self.inv_sigma_sq * z * z)
        lps.fix_external_set_energy_peratom("ext", E)
        Etot = MPI.COMM_WORLD.allreduce(E.sum())
        lps.fix_external_set_energy_global("ext", Etot)
        # --- corresponding force:
        f.fill(0.)
        f[:, 2] = E * self.inv_sigma_sq * z
        # --- corresponding virial:
        virial_mat = pos.T @ f  # 3 x 3 matrix
        virial_mat = 0.5*(virial_mat + virial_mat.T)  # symmetric tensor
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, virial_mat)
        vtot = [
            virial_mat[0, 0], virial_mat[1, 1], virial_mat[2, 2],  # xx, yy, zz
            virial_mat[0, 1], virial_mat[0, 2], virial_mat[1, 2],  # xy, xz, yz
        ]
        lps.fix_external_set_virial_global("ext", vtot)

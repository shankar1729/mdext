import numpy as np
from mdext import MPI
from abc import ABC, abstractmethod


class Potential(ABC):
    """Base class for all external potentials."""
    def __init__(self, geometry: str):
        assert geometry in {'planar', 'cylindrical', 'spherical'}
        self.geometry = geometry
    
    @abstractmethod
    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Calculate external forces in `f` and return corresponding per-atom energies,
        given the positions `pos` (wrapped to unit cell centered at zero).
        """
        
    def __call__(self, lps, ntimestep, nlocal, tag, x, f) -> None:
        """
        Callback function to add force, energy and virials for external potential.
        Handles total energy and virial calculation using forces from `self.compute`.
        """
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        
        # Wrap positions periodically:
        pos = x/L  # to fractional coordinates
        pos -= np.floor(0.5 + pos)  # wrap to [-0.5, 0.5)
        pos *= L  # back to Cartesian coordinates
        
        # Get energies and forces (from derived class):
        f.fill(0.)
        E = self.compute(pos, f)
        
        # Compute total energy and virial:
        Etot = MPI.COMM_WORLD.allreduce(E.sum())
        virial_mat = pos.T @ f  # 3 x 3 matrix
        virial_mat = 0.5*(virial_mat + virial_mat.T)  # symmetric tensor
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, virial_mat)
        
        # Set energy and virial globals (forces already set in-place):
        lps.fix_external_set_energy_peratom("ext", E)
        lps.fix_external_set_energy_global("ext", Etot)
        vtot = [
            virial_mat[0, 0], virial_mat[1, 1], virial_mat[2, 2],  # xx, yy, zz
            virial_mat[0, 1], virial_mat[0, 2], virial_mat[1, 2],  # xy, xz, yz
        ]
        lps.fix_external_set_virial_global("ext", vtot)


class PlanarGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up planar Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("planar")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        z = pos[:, 2]
        E = self.U0 * np.exp(-0.5 * self.inv_sigma_sq * z * z)
        f[:, 2] = E * self.inv_sigma_sq * z
        return E


class CylindricalGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up cylindrical Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("cylindrical")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        xy = pos[:, :2]
        rho_sq = (xy ** 2).sum(axis=1)
        E = self.U0 * np.exp(-0.5 * self.inv_sigma_sq * rho_sq)
        f[:, :2] = (E * self.inv_sigma_sq)[:, None] * xy
        return E


class SphericalGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up spherical Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("spherical")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        r_sq = (pos ** 2).sum(axis=1)
        E = self.U0 * np.exp(-0.5 * self.inv_sigma_sq * r_sq)
        f[:] = (E * self.inv_sigma_sq)[:, None] * pos
        return E

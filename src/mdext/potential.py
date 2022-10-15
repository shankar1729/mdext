import numpy as np
from mdext import MPI
from abc import ABC, abstractmethod
from .histogram import Histogram


class Potential(ABC):
    """Base class for all external potentials."""
    def __init__(self, geometry: str):
        assert geometry in {'planar', 'cylindrical', 'spherical'}
        self.geometry = geometry
    
    def initialize_histogram(self, lps, dr: float, n_atom_types: int) -> None:
        """Initialize density collection with same 1D geometry as applied potential."""
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        dim_sel = {
            'planar': 2, 'cylindrical': slice(2), 'spherical': slice(None),
        }[self.geometry]
        r_max = 0.5 * L[dim_sel].min()  # in-radius in relevant dimensions
        self.hist = Histogram(0.0, r_max, dr, n_atom_types)  # each thermo
        self.dr = dr
        self.r = self.hist.bins
        # --- bin-dependent weight of histograms:
        w_intervals = np.zeros(len(self.r) + 1)  # with extra intervals to left & right
        if self.geometry == "planar":
            w_intervals[1:-1] = 2 * (self.r[1:] - self.r[:-1])
        elif self.geometry == "cylindrical":
            w_intervals[1:-1] = np.pi * (self.r[1:] ** 2 - self.r[:-1] ** 2)
        elif self.geometry == "spherical":
            w_intervals[1:-1] = (4 * np.pi / 3) * (self.r[1:] ** 3 - self.r[:-1] ** 3)
        else:
            raise KeyError(f"Invalid potential {self.geometry = }")
        self.w_bins = dr / (0.5 * (w_intervals[:-1] + w_intervals[1:]))
        self.n_steps = 0

    def reset_stats(self):
        self.hist.reset()
        self.n_steps = 0

    @property
    def density(self):
        """Densities collected with same 1D geometry as applied potential."""
        result = self.hist.hist * self.w_bins[:, None] / self.n_steps
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, result)
        return result
        
    @abstractmethod
    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Calculate external forces in `f` and return corresponding per-atom energies,
        given the positions `pos` (wrapped to unit cell centered at zero).
        """

    @abstractmethod
    def get_potential(self, r: np.ndarray) -> np.ndarray:
        """
        Return potential as a function of position `r`.
        This is used to store the applied potential with the measured density response.
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

        # Collect densities for this step:
        if self.geometry == "planar":
            r = abs(pos[:, 2])  # coordinate being collected
            w_box = 1.0 / (L[0] * L[1])  # box-size dependent weight
        elif self.geometry == "cylindrical":
            r = np.linalg.norm(pos[:, :2], axis=1)  # coordinate being collected
            w_box = 1.0 / L[2]  # box-size dependent weight
        elif self.geometry == "spherical":
            r = np.linalg.norm(pos, axis=1)  # coordinate being collected
            w_box = 1.0  # box-size dependent weight
        else:
            raise KeyError(f"Invalid potential {geometry = }")
        
        weights = np.zeros((nlocal, self.hist.n_w))
        types = lps.numpy.extract_atom("type")
        for i_type in range(self.hist.n_w):
            weights[(types == i_type + 1), i_type] = w_box
        self.hist.add_events(r, weights)
        self.n_steps += 1


class PlanarGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up planar Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("planar")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        z = pos[:, 2]
        E = self.get_potential(z)
        f[:, 2] = E * self.inv_sigma_sq * z
        return E
    
    def get_potential(self, r: np.ndarray) -> np.ndarray:
        return self.U0 * np.exp(-0.5 * self.inv_sigma_sq * r * r)


class CylindricalGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up cylindrical Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("cylindrical")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        xy = pos[:, :2]
        E = self.get_potential(np.linalg.norm(xy, axis=1))
        f[:, :2] = (E * self.inv_sigma_sq)[:, None] * xy
        return E

    def get_potential(self, r: np.ndarray) -> np.ndarray:
        return self.U0 * np.exp(-0.5 * self.inv_sigma_sq * r * r)


class SphericalGaussian(Potential):
    
    def __init__(self, U0: float, sigma: float) -> None:
        """Set up spherical Gaussian potential with peak `U0` and width `sigma`."""
        super().__init__("spherical")
        self.U0 = U0
        self.sigma = sigma
        self.inv_sigma_sq = 1./(sigma**2)

    def compute(self, pos: np.ndarray, f: np.ndarray) -> np.ndarray:
        E = self.get_potential(np.linalg.norm(pos, axis=1))
        f[:] = (E * self.inv_sigma_sq)[:, None] * pos
        return E

    def get_potential(self, r: np.ndarray) -> np.ndarray:
        return self.U0 * np.exp(-0.5 * self.inv_sigma_sq * r * r)

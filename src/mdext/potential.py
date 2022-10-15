from lammps import lammps
import numpy as np
from mdext import MPI
from .geometry import GeometryType
from .histogram import Histogram
from typing import Callable, Tuple


Potential = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
"""Return potential and its derivative, given square of 1D coordinate as input.
Note that the derivative is also with respect to the squared-coordinate."""


class Gaussian:
    def __init__(self, U0: float, sigma: float) -> None:
        """Gaussian potential with peak `U0` and width `sigma`."""
        self.U0 = U0
        self.sigma = sigma
        self.mhalf_inv_sigma_sq = -0.5 / (sigma ** 2)

    def __call__(self, r_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E = self.U0 * np.exp(self.mhalf_inv_sigma_sq * r_sq)
        r_sq_grad = self.mhalf_inv_sigma_sq * E
        return E, r_sq_grad


class ForceCallback:
    """Force callback object to apply external forces and collect densities."""

    def __init__(
        self,
        *,
        potential: Potential,
        geometry_type: GeometryType,
        lps: lammps,
        dr: float,
        n_atom_types: int,
    ):
        self.potential = potential
        self.geometry_type = geometry_type
    
        # Initialize density collection with same 1D geometry as applied potential
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        self.hist = Histogram(0.0, geometry_type.r_max(L), dr, n_atom_types)
        self.dr = dr
        self.r = self.hist.bins

        # Calculate bin-dependent weight of histograms (eg. 2 pi r for cylindrical):
        volumes = geometry_type.volume(self.r)
        w_intervals = np.zeros(len(self.r) + 1)  # with extra intervals to left & right
        w_intervals[1 : -1] = volumes[1:] - volumes[:-1]
        self.w_bins = dr / (0.5 * (w_intervals[:-1] + w_intervals[1:]))
        self.n_steps = 0

    def reset_stats(self) -> None:
        self.hist.reset()
        self.n_steps = 0

    @property
    def density(self) -> np.ndarray:
        """Densities collected with same 1D geometry as applied potential."""
        result = self.hist.hist * self.w_bins[:, None] / self.n_steps
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, result)
        return result
        
    def __call__(
        self,
        lps: lammps,
        ntimestep: int,
        nlocal: int,
        tag: np.ndarray,
        x: np.ndarray,
        f: np.ndarray
    ) -> None:
        """
        Callback function to add force, energy and virials for external potential.
        """
        boxlo, boxhi = lps.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        
        # Wrap positions periodically:
        pos = x/L  # to fractional coordinates
        pos -= np.floor(0.5 + pos)  # wrap to [-0.5, 0.5)
        pos *= L  # back to Cartesian coordinates
        
        # Get energies and forces (from derived class):
        f.fill(0.)
        r_sq = self.geometry_type.r_sq(pos)
        E, r_sq_grad = self.potential(r_sq)
        self.geometry_type.set_force(pos, r_sq_grad, f)
        
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
        inv_perpendicular_volume = 1.0 / self.geometry_type.perpendicular_volume(L)
        weights = np.zeros((nlocal, self.hist.n_w))
        types = lps.numpy.extract_atom("type")
        for i_type in range(self.hist.n_w):
            weights[(types == i_type + 1), i_type] = inv_perpendicular_volume
        self.hist.add_events(np.sqrt(r_sq), weights)
        self.n_steps += 1

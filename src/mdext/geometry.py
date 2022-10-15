import numpy as np
from typing import TypeVar, Protocol, Sequence


GeometryType = TypeVar("GeometryType", bound="Geometry")


class Geometry(Protocol):
    """Defines a 1D geometry for applied potentials and collected densities."""

    @staticmethod
    def r_sq(pos: np.ndarray) -> np.ndarray:
        """Get the square of the 1D coordinates from the 3D coordinates, `pos`.
        Here, `pos` will be N x 3 Cartesian coordinates wrapped to the unit cell
        centered at the origin, and the output should be an array of length N."""

    @staticmethod
    def set_force(pos: np.ndarray, r_sq_grad: np.ndarray, f: np.ndarray) -> None:
        """Set forces in `f`, given positions `pos` and the derivative of energy
        with respect to r_sq in `r_sq_grad`. Essentially, this is the gradient
        propagation routine corresponding to `r_sq`."""

    @staticmethod
    def r_max(L: Sequence[float]) -> float:
        """Get maximum radius for density collection, given box-size `L`."""

    @staticmethod
    def perpendicular_volume(L: Sequence[float]) -> float:
        """Get volume perpendicular to the space spanned by the 1D coordinate.
        This amounts to 1, Lz and Axy in spherical, cylindirical and planar
        geometries respectively. Here, `L` is the box size."""

    @staticmethod
    def volume(r: np.ndarray) -> np.ndarray:
        """Get cumulative volume upto each entry of `r`."""


class Planar:
    @staticmethod
    def r_sq(pos: np.ndarray) -> np.ndarray:
        return pos[:, 2] ** 2

    @staticmethod
    def set_force(pos: np.ndarray, r_sq_grad: np.ndarray, f: np.ndarray) -> None:
        f[:, 2] = -2.0 * r_sq_grad * pos[:, 2]
    
    @staticmethod
    def r_max(L: Sequence[float]) -> float:
        return 0.5 * L[2]

    @staticmethod
    def perpendicular_volume(L: Sequence[float]) -> float:
        return L[0] * L[1]

    @staticmethod
    def volume(r: np.ndarray) -> np.ndarray:
        return 2 * r


class Cylindrical:
    @staticmethod
    def r_sq(pos: np.ndarray) -> np.ndarray:
        return (pos[:, :2] ** 2).sum(axis=1)

    @staticmethod
    def set_force(pos: np.ndarray, r_sq_grad: np.ndarray, f: np.ndarray) -> None:
        f[:, :2] = (-2.0 * r_sq_grad)[:, None] * pos[:, :2]

    @staticmethod
    def r_max(L: Sequence[float]) -> float:
        return 0.5 * min(L[0], L[1])

    @staticmethod
    def perpendicular_volume(L: Sequence[float]) -> float:
        return L[2]

    @staticmethod
    def volume(r: np.ndarray) -> np.ndarray:
        return np.pi * (r ** 2)


class Spherical:
    @staticmethod
    def r_sq(pos: np.ndarray) -> np.ndarray:
        return (pos ** 2).sum(axis=1)

    @staticmethod
    def set_force(pos: np.ndarray, r_sq_grad: np.ndarray, f: np.ndarray) -> None:
        f[:] = (-2.0 * r_sq_grad)[:, None] * pos

    @staticmethod
    def r_max(L: Sequence[float]) -> float:
        return 0.5 * min(L)

    @staticmethod
    def perpendicular_volume(L: Sequence[float]) -> float:
        return 1.0

    @staticmethod
    def volume(r: np.ndarray) -> np.ndarray:
        return (4 * np.pi / 3) * (r ** 3)

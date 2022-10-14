import numpy as np


class Histogram:
    """Linear-interpolated weighted histogram with efficient multi-channel support.
    Operation is functionally equivalent to `numpy.histogram` with density = True and
    with weights, but operates on several weights at once, and linearly interpolates
    results within each bin to better sample probability density with fewer bins."""

    def __init__(self, x_min: float, x_max: float, dx: float, n_w: int) -> None:
        """Initiate histogram with bins `arange(x_min, x_max, dx)` with n_w weight
        channels."""
        self.x_min = x_min
        self.x_max = x_max
        self.dx_inv = 1./dx
        self.n_w = n_w
        self.bins = np.arange(x_min, x_max, dx)
        self.hist = np.zeros((len(self.bins), n_w))
        self.n_intervals = len(self.bins) - 1
    
    def add_events(self, x: np.ndarray, w: np.ndarray) -> None:
        """Add contributions from `x` (array of length N)
        with weights `w` (N x n_w array)."""
        x_frac = (x - self.x_min) * self.dx_inv  # fractional coordinate
        i = np.floor(x_frac).astype(int)
        # Select range of collection:
        sel = np.where(np.logical_and(i >= 0, i < self.n_intervals))
        i = i[sel]
        t = (x_frac[sel] - i)[:, None]  # to broadcast with n_w weights below
        w_by_dx = w[sel] * self.dx_inv
        # Histogram:
        np.add.at(self.hist, i, (1.-t) * w_by_dx)
        np.add.at(self.hist, i + 1, t * w_by_dx)
    
    def reset(self) -> None:
        """Reset counts to zero."""
        self.hist.fill(0.)

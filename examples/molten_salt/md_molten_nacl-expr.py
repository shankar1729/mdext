# Simulation script for NaCl in external potentials
# this script can be launched in a jobfile (perturb.job) with options for different geometry types and external potential parameters

from lammps import lammps, PyLammps, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR
from mpi4py import MPI
import numpy as np
import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="add external potential to molten NaCl")
parser.add_argument('-A', action='store', dest='A') # amplitude (magnitude) of external potential 
parser.add_argument('-s', action='store', dest='s') # sigma, inverse width of external potential
parser.add_argument('-T', action='store', dest='T') # temperature
parser.add_argument('-type',action='store', dest='type') # type of external Gaussian potential - one of 'att', 'rep' (attractive or repulsive)
parser.add_argument('-geom',action='store', dest='geom') # type of geometry - one of 'p', 'r', 'c' (planar, radial, cylindrical)
args = parser.parse_args()
log = logging.getLogger("md")  # use instead of print for mpi-head-only output
md = None  # global object used in lammps callback (created in main)


def main() -> None:
    # Initialize log:
    is_head = (MPI.COMM_WORLD.rank == 0)
    log.setLevel(logging.INFO if is_head else logging.WARNING)
    log.addHandler(logging.StreamHandler(sys.stderr))

    # Current simulation parameters:
    T = float(args.T)
    seed = np.random.randint(1,100)
    if args.type == 'rep':
        U0 = float(args.A)   # Amplitude of the external potential in eV
    elif args.type == 'att':
        U0 = -1.*float(args.A)
    sigma = float(args.s) # inverse angstroms
    boxsize = 8 # number of supercells in one dimension
    lc = 5.627 # NaCl lattice constant in angstroms around room temperature

    # Initialize and run simulation:
    global md
    # initialize sim params, potential etc - U0, sigma only get called later
    md = MD(T=T, boxsize=boxsize, lc=lc, seed=seed, U0=U0, sigma=sigma, geom=args.geom) 
    # call functions to run individual stages of dynamics - start from an equilibrated NaCl melt instead of a random config
    md.melt(5000,"lattice melt") 
    md.quench(2500, "melt quench", T)
    md.npt_equilibrate(20000, "NPT", 1., T) 
    md.reset_stats()
    md.perturb(100000,"allowing applied perturbation to equilibrate....", T) # callback for external forces should only be applied here
    #md.reset_stats()
    #md.perturb(10000,"collecting density data", T)
    
    # Plot:
    if is_head:
        log.info(f"{md.n_steps_collected = }")
        U = U0 * np.exp(-0.5 * (md.z / sigma)**2)  # potential
        kB = 1.987204259E-3  # in kcal/mol/K
        # !!TODO: add histograms in all 3 dimensions 
        plt.plot(md.z, md.density / md.n_steps_collected, label=f"Ext pot: {args.geom}")
        plt.xlabel("z (A)")
        plt.ylabel("Density (??)")
        plt.legend()
        plt.savefig(f'nacl-{args.geom}_U0-{U0}_sig-{sigma}.png')
    
class MD:
    def __init__(self, *, T: float, boxsize: int, lc: float, seed: int, U0: float, sigma: float, geom: str) -> None:
        '''
        don't run any dynamics in __init__, set up system and potential
        run minimization
        run stages of dynamics in separate individual functions
        '''
        np.random.seed(seed)

        # Set up LAMMPS instance:
        lps = lammps()
        os.environ["OMP_NUM_THREADS"] = "1"  # run single-threaded
        lmp = PyLammps(ptr=lps)

        # Global settings:
        lmp.units("metal") # eV, Angstrom
        lmp.dimension("3")
        lmp.atom_style("charge")
        lmp.boundary("p p p")

        # Construct simulation box:
        L = [boxsize*lc, boxsize*lc, boxsize*lc]  # overall box dimensions
        n_atom_types = 2

        # anion lattice
        lmp.lattice(f"fcc {lc}")
        lmp.region(
            f"box block -{boxsize/2} {boxsize/2} -{boxsize/2} {boxsize/2} -{boxsize/2} {boxsize/2}"
        )
        lmp.create_box( f"{n_atom_types} box")
        lmp.create_atoms("1 box")

        # cation lattice
        lmp.lattice(f"fcc {lc} origin 0.5 0. 0.")
        lmp.create_atoms("2 box")

        n_atoms = 8*(boxsize**3) # 8 atoms per rocksalt unit cell
        lmp.mass("1 35.453")
        lmp.mass("2 22.9898")

        # Fumi-Tosi NaCl
        log.info("Setting force field parameters")
        lmp.set(f"type 1 charge -1")
        lmp.set(f"type 2 charge +1")
        #lmp.pair_style("born/coul/dsf 0.25 9.0")
        lmp.pair_style("born/coul/long 9.0")
        lmp.pair_coeff("1 1 0.158221 0.327 3.170 75.0544 -150.7325")  # Cl-Cl
        lmp.pair_coeff("2 2 0.2637 0.317 2.340 1.048553 -0.49935")  # Cl-Na
        lmp.pair_coeff("1 2 0.21096 0.317 2.755 6.99055303 -8.6757") # Na-Na
        lmp.kspace_style("ewald 1e-5")

        # Initial minimize:
        log.info("Minimizing initial structure")
        lmp.thermo("100")
        lmp.thermo_style("custom step temp ke pe density press fmax") 
        # use a separate log and dump file for each stage of dynamics

        lmp.minimize("1E-4 1E-6 10000 100000")
        # --- reset neighbor and timesteps
        lmp.neighbor("2. bin")
        lmp.neigh_modify("exclude none every 1 check yes")
        lmp.timestep("0.001")  # in picoseconds

        # Initialize velocities:
        lmp.velocity(f"all create {T} {seed} dist gaussian loop local")
        

        # Store instance variables needed later:
        self.geom = geom
        self.is_head = (MPI.COMM_WORLD.rank == 0)
        self.lmp = lmp
        self.lps = lps
        self.T = T
        self.U0 = U0
        self.sigma = sigma
        self.L = L
        self.n_atoms = n_atoms
        # --- setup histograms:
        self.dz = 0.05
        self.hist = Histogram(-L[2]/2, L[2]/2, self.dz, 1)  # used in each thermo
        self.z = self.hist.bins
        self.density = np.zeros((len(self.z), self.hist.n_w))
        self.n_steps_collected = 0  # number of steps collected in density

    def reset_stats(self) -> None:
        self.lmp.reset_timestep(0)
        self.n_steps_collected = 0
        self.density.fill(0.)


    # don't use this function, call lmp.run for different types of dynamics in functions below
    def run(self, n_steps: int, name: str) -> None:
        log.info(f"Starting {name}")
        self.lmp.run(n_steps)
        log.info(f"Completed {name}")

    def melt(self, n_steps: int, name: str) -> None:
        log.info("Melting lattice")
        self.lmp.log("md_melt.log")
        self.lmp.dump("0 all custom 100 md_melt.dump id type x y z")
        self.lmp.fix(f"melt all nvt temp 3000. 3000. $(20*dt)")
        self.lmp.run(n_steps)
        self.lmp.unfix("melt")
        self.lmp.undump("0")

    def quench(self, n_steps: int, name: str, T: float) -> None:
        log.info("Quenching melt")
        self.lmp.log("md_quench.log")
        self.lmp.dump("0 all custom 100 md_quench.dump id type x y z")
        self.lmp.fix(f"quench all nvt temp 3000. {T} $(20*dt)")
        self.lmp.run(n_steps)
        self.lmp.unfix("quench")
        self.lmp.undump("0")

    def npt_equilibrate(self ,n_steps: int, name: str, P: float, T: float):
        log.info("Equilibrating system at constant pressure and temperature")
        self.lmp.log("md_npt_equilibrate.log")
        self.lmp.dump("0 all custom 100 md_npt_equilibrate.dump id type x y z")
        self.lmp.fix(f"npt_eq all npt temp {T} {T} $(20*dt) iso {P} {P} $(500*dt)")
        self.lmp.run(n_steps)
        self.lmp.unfix("npt_eq")
        self.lmp.undump("0")

    def nvt_equilibrate(self, n_steps: int, name: str, T: float):
        log.info("Equilibrating system at constant volume and temperature")
        self.lmp.log("md_nvt_equilibrate.log")
        self.lmp.dump("0 all custom 100 md_npt_equilibrate.dump id type x y z")
        self.lmp.fix(f"nvt_eq all nvt temp {T} {T} $(20*dt)")
        self.lmp.run(n_steps)
        self.lmp.unfix("nvt_eq")
        self.lmp.undump("0")

    # callback function for external potential ONLY called here
    # perturbations are NVT by default
    def perturb(self, n_steps: int, name: str, T: float): 
        log.info("Adding external gaussian potential in specified geometry at constant volume")
        self.lmp.log(f'nacl-{self.geom}_U0-{self.U0}_sig-{self.sigma}.log')
        self.lmp.dump(f"0 all custom 100 nacl-{self.geom}_U0-{self.U0}_sig-{self.sigma}.dump id type x y z")
        self.lmp.fix("ext all external pf/callback 1 1") 
        self.lps.set_fix_external_callback("ext", self, self.lps)
        self.lmp.fix(f"nvt_int all nvt temp {T} {T} $(20*dt)")
        self.lmp.run(n_steps)
        self.lmp.unfix("ext")
        self.lmp.unfix("nvt_int")
        self.lmp.undump("0")
        

    
    def __call__(self, lps, ntimestep, nlocal, tag, x, f) -> None:
        """Callback function to add external potential and collect data."""

        # Following the same logic in the mdext/geometry.py and mdext/potential.py files to set energies and forces as appopriate for different geometries
        # this involves correctly defining r_sq and r_sq_grad based on the geometry type

        inv_sigma_sq = 1./(self.sigma**2)

        # first take care of PBCs
        pos = x/self.L # fractional coordinates
        pos -= np.floor(0.5 + pos) # periodic wrap
        pos *= self.L # back to cartesian 

        # planar
        if self.geom == 'p':
            r_sq = pos[:, 2] ** 2
            # --- energy: 
            E = self.U0 * np.exp(-0.5 * inv_sigma_sq * r_sq)
            lps.fix_external_set_energy_peratom("ext", E)
            Etot = MPI.COMM_WORLD.allreduce(E.sum())
            lps.fix_external_set_energy_global("ext", Etot)
            # --- forces:
            f.fill(0.)
            r_sq_grad = -0.5 * inv_sigma_sq * E
            f[:, 2] = -2.0* r_sq_grad * pos[:, 2]

        # cylindrical
        elif self.geom == 'c':
            r_sq = (pos[:, :2] ** 2).sum(axis=1)
            # --- energy: 
            E = self.U0 * np.exp(-0.5 * inv_sigma_sq * r_sq)
            lps.fix_external_set_energy_peratom("ext", E)
            Etot = MPI.COMM_WORLD.allreduce(E.sum())
            lps.fix_external_set_energy_global("ext", Etot)
            # --- forces:
            f.fill(0.)
            r_sq_grad = -0.5 * inv_sigma_sq * E
            f[:, :2] = (-2.0 * r_sq_grad)[:, None] * pos[:, :2]

        # radial/spherical
        elif self.geom == 'r':
            r_sq = (pos ** 2).sum(axis=1)
            # --- energy: 
            E = self.U0 * np.exp(-0.5 * inv_sigma_sq * r_sq)
            lps.fix_external_set_energy_peratom("ext", E)
            Etot = MPI.COMM_WORLD.allreduce(E.sum())
            lps.fix_external_set_energy_global("ext", Etot)
            # --- forces:
            f.fill(0.)
            r_sq_grad = -0.5 * inv_sigma_sq * E
            f[:] = (-2.0 * r_sq_grad)[:, None] * pos

        
        # Collect density
        # TODO: add histograms along x and y
        invA = 1./(self.L[0] * self.L[1])
        weights = np.zeros((nlocal, 1))
        types = lps.numpy.extract_atom("type")[:nlocal]
        for i_type in range(self.hist.n_w):
            weights[(types == i_type + 1), i_type] = invA  # for each atom type
        self.hist.reset()
        # add options to collect x and y distributions too
        self.hist.add_events(pos[:,2], weights) 
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, self.hist.hist)
        # --- fix boundary periodicity
        hist_end = self.hist.hist[0] + self.hist.hist[-1]
        self.hist.hist[0] = hist_end
        self.hist.hist[-1] = hist_end
        # --- accumulate
        self.density += self.hist.hist
        self.n_steps_collected += 1


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


if __name__ == "__main__":
    main()





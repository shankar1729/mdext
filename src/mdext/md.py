from lammps import lammps, PyLammps, LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR
import mdext
from mdext import MPI, potential
from .histogram import Histogram
import numpy as np
import os
import time
import logging
import sys


log: logging.Logger = logging.getLogger("mdext")  #: Log for the mdext module
log.setLevel(logging.WARNING if MPI.COMM_WORLD.rank else logging.INFO)
log.addHandler(logging.StreamHandler(sys.stderr))  # because PyLAMMPS captures stdout

    
class MD:
    
    def __init__(self, *, T: float, seed: int, U0: float, sigma: float) -> None:
        np.random.seed(seed)

        # Set up LAMMPS instance:
        lps = lammps()
        os.environ["OMP_NUM_THREADS"] = "1"  # run single-threaded
        lmp = PyLammps(ptr=lps)
        self.t_start = time.time()

        # Global settings:
        lmp.units("real") # kcal/mol, Angstrom
        lmp.dimension("3")
        lmp.atom_style("full")
        lmp.boundary("p p p")

        # Construct simulation box:
        L = np.array([30., 30., 30.])  # overall box dimensions
        n_atom_types = 1 

        lmp.region(
            f"sim_box block -{L[0]/2} {L[0]/2} -{L[1]/2} {L[1]/2} -{L[2]/2} {L[2]/2}"
            " units box"
        )
        lmp.create_box( f"{n_atom_types} sim_box")
        n_atoms = 900  # roughly the bulk density of water
        lmp.create_atoms(f"1 random {n_atoms} {seed} sim_box")
        lmp.mass("1 15.9994")

        # Interaction potential:
        lmp.pair_style("lj/cut 10")
        lmp.pair_coeff("1 1 1.0 2.97")  # Adjusted to get water bulk density
        lmp.pair_modify("tail yes")

        # Initial minimize:
        log.info("Minimizing initial structure")
        #lmp.neigh_modify("exclude molecule/intra all")
        n_thermo = 10
        lmp.thermo(n_thermo)
        lmp.thermo_style("custom step temp press pe")
        lmp.minimize("1E-4 1E-6 10000 100000")
        # --- reset neighbor and timesteps
        lmp.reset_timestep("0")
        lmp.neighbor("2. bin")
        lmp.neigh_modify("exclude none every 1 check yes")
        lmp.timestep("2.")  # in femtoseconds

        # Set up for dynamics:
        lmp.velocity(f"all create {T} {seed} dist gaussian loop local")
        #lmp.fix(f"Ensemble all nvt temp {T} {T} 100")  # Td = 100 fs
        lmp.fix(f"Ensemble all npt temp {T} {T} 100 iso 1 1 100")  # Td = 100 fs
        
        # --- external fix
        global extpot
        extpot = potential.Planar(U0, sigma)
        lmp.fix("ext all external pf/callback 1 1")
        lps.set_fix_external_callback("ext", extpot, lps)
        
        # Save required simulation parameters in instance:
        self.is_head = (MPI.COMM_WORLD.rank == 0)
        self.lmp = lmp
        self.T = T
        self.U0 = U0
        self.sigma = sigma
        self.n_thermo = n_thermo  # number of steps between thermo (10 x 2 fs = 20 fs)
        self.thermo_per_cycle = 500  # number of thermo per cycle (50 x 20 fs = 10 ps)
        self.steps_per_cycle = n_thermo * self.thermo_per_cycle
        self.i_thermo = -1  # index of current thermo entry within cycle
        self.i_cycle = 0  # index of current cycle
        self.cycle_stats = np.zeros(4)  # cumulative T, P, PE, vol
        # --- setup histograms:
        self.dz = 0.05
        self.hist = Histogram(-L[2]/2, L[2]/2, self.dz, 1)  # used in each thermo
        self.z = self.hist.bins
        self.cycle_density = np.zeros((len(self.z), self.hist.n_w))  # results within a cycle
        self.density = np.zeros_like(self.cycle_density)  # cumulative results

    def reset_stats(self) -> None:
        """Reset counts / histograms."""
        self.lmp.reset_timestep(0)
        self.i_thermo = -1
        self.i_cycle = 0
        self.cycle_density.fill(0.)
        self.density.fill(0.)

    def setup_thermo_callback(self) -> None:
        """Setup data collection at every thermo step."""
        self.lmp.python("md input 1 SELF return v_thermo_callback format pf exists")
        self.lmp.variable("thermo_callback python md")
        self.lmp.thermo_style("custom step temp press pe vol v_thermo_callback")

    def run(self, n_cycles: int, run_name: str, density_file: str = "") -> None:
        """Run `n_cycles` cycles of `steps_per_cycle` each.
        Use `run_name` to report the start and end of the run (to ease parsing the log).
        If `density_file` is specified, save densities after every cycle."""
        log.info(f"Starting {run_name}")
        log.info("Cycle  T[K]   P[bar]  PE[kcal/mol] vol[A^3] t_cpu[s]")
        for i_cycle in range(n_cycles):
            self.lmp.run(self.steps_per_cycle)
            if density_file:
                self.save_density(density_file)
        log.info(f"Completed {run_name}")

    def __call__(self, lmp_ptr) -> float:
        """Callback function invoked during each thermo cycle to collect densities."""
        if self.i_thermo == -1:
            # Ignore first step redundant with previous cycle:
            self.i_thermo = 0
            return 0.
        
        # Get relevant data from LAMMPS:
        lmp = lammps(ptr=lmp_ptr)
        types = lmp.numpy.extract_atom("type")
        pos = lmp.numpy.extract_atom("x")
        boxlo, boxhi = lmp.extract_box()[:2]
        L = np.array(boxhi) - np.array(boxlo)
        
        # Wrap positions periodically:
        pos = pos/L  # to fractional coordinates
        pos -= np.floor(0.5 + pos)  # wrap to [-0.5, 0.5)
        pos *= L  # back to Cartesian coordinates

        # Collect densities for this step:
        invA = 1./(L[0] * L[1])
        weights = np.zeros((len(types), self.hist.n_w))
        for i_type in range(self.hist.n_w):
            weights[(types == i_type + 1), i_type] = invA
        self.hist.reset()
        self.hist.add_events(pos[:, 2], weights)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, self.hist.hist)
        
        # Collect results over cycle:
        self.cycle_density += self.hist.hist
        self.cycle_stats += np.array((
            lmp.get_thermo("temp"),
            lmp.get_thermo("press"),
            lmp.get_thermo("pe"),
            L.prod(),
        ))
        self.i_thermo += 1

        # Report when cycle complete:
        if self.i_thermo == self.thermo_per_cycle:
            cycle_norm = 1. / self.thermo_per_cycle
            T, P, PE, vol = self.cycle_stats * cycle_norm
            t_cpu = time.time() - self.t_start
            self.density += self.cycle_density * cycle_norm
            self.i_cycle += 1
            log.info(
                f"{self.i_cycle:^5d} {T:7.3f} {P:7.1f} {PE:^12.3f} "
                f"{vol:^8.1f} {t_cpu:7.1f}"
            )
            # Reset within-cycle quantities:
            self.cycle_density.fill(0.)
            self.cycle_stats.fill(0.)
            self.i_thermo = -1
        
        return 0.

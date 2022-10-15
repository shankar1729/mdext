from lammps import lammps, PyLammps
import mdext
from mdext import MPI, potential
from .histogram import Histogram
import numpy as np
import os
import time
import logging
import sys
from typing import Callable


log: logging.Logger = logging.getLogger("mdext")  #: Log for the mdext module
log.setLevel(logging.WARNING if MPI.COMM_WORLD.rank else logging.INFO)
log.addHandler(logging.StreamHandler(sys.stderr))  # because PyLAMMPS captures stdout

thermo_callback = None  #: imported into __main__ by lammps python command

    
class MD:
    
    def __init__(
        self,
        *,
        setup: Callable[[PyLammps, int], None],
        T: float,
        seed: int,
        U0: float,
        sigma: float,
        n_thermo: int = 10,
    ) -> None:
        np.random.seed(seed)

        # Set up LAMMPS instance:
        lps = lammps()
        os.environ["OMP_NUM_THREADS"] = "1"  # run single-threaded
        lmp = PyLammps(ptr=lps)
        self.t_start = time.time()

        # Global settings:
        lmp.units("real") # kcal/mol, Angstrom
        lmp.dimension("3")
        lmp.boundary("p p p")
        
        # Set up initial atomic configuration and interaction potential:
        setup(lmp, seed)

        # Prepare for dynamics:
        lmp.reset_timestep("0")
        lmp.timestep("2.")  # in femtoseconds
        lmp.velocity(f"all create {T} {seed} dist gaussian loop local")
        #lmp.fix(f"Ensemble all nvt temp {T} {T} 100")  # Td = 100 fs
        lmp.fix(f"Ensemble all npt temp {T} {T} 100 iso 1 1 100")  # Td = 100 fs
        # --- setup thermo callback
        lmp.thermo(n_thermo)
        mdext.md.thermo_callback = self
        lmp.python(
            "thermo_callback input 1 SELF return v_thermo_callback format pf here"
            " 'from mdext.md import thermo_callback'"
        )
        lmp.variable("thermo_callback python thermo_callback")
        lmp.thermo_style("custom step temp press pe vol v_thermo_callback")
        # --- set up external force callback
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
        boxlo, boxhi = lps.extract_box()[:2]
        self.hist = Histogram(boxlo[2], boxhi[2], self.dz, 1)  # used in each thermo
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

from lammps import lammps, PyLammps
import mdext
from mdext import MPI, potential
from .histogram import Histogram
import numpy as np
import os
import time
import logging
import sys
from typing import Callable, Optional


log: logging.Logger = logging.getLogger("mdext")  #: Log for the mdext module
log.setLevel(logging.WARNING if MPI.COMM_WORLD.rank else logging.INFO)
log.addHandler(logging.StreamHandler(sys.stderr))  # because PyLAMMPS captures stdout

thermo_callback = None  #: imported into __main__ by lammps python command

unit_names = {
    'real': {
        'mass': 'amu',
        'distance': 'A',
        'time': 'fs',
        'energy': 'kcal/mol',
        'temperature': 'K',
        'pressure': 'atm',
    },
    'metal': {
        'mass': 'amu',
        'distance': 'A',
        'time': 'ps',
        'energy': 'eV',
        'temperature': 'K',
        'pressure': 'bar',
    },
    'electron': {
        'mass': 'amu',
        'distance': 'a0',
        'time': 'fs',
        'energy': 'Eh',
        'temperature': 'K',
        'pressure': 'Pa',
    },
}  #: unit names (for logging)

    
class MD:
    
    def __init__(
        self,
        *,
        setup: Callable[[PyLammps, int], None],
        T: float,
        P: Optional[float],
        seed: int,
        potential: potential.Potential,
        units: str = "real",
        timestep: float = 2.0,
        n_thermo: int = 10,
        thermo_per_cycle: int = 500,
        dr: float = 0.05,
        Tdamp: float = 100.0,
        Pdamp: float = 100.0,
    ) -> None:
        """
        Parameters
        ----------
        setup
            Callable with signature `setup(lmp, seed)` that creates initial atomic
            configuration and sets up the interaction potential / force fields.
            This could load a LAMMPS data file, or use LAMMPS box and random atom
            creation commands. If starting with a random configuration, this should
            also ideally invoke minimize to ensure a reasonable starting point.
            (Don't need to equlibrate here; that can be done using `MD.run` later.)
        T
            Temperature in LAMMPS temperature units.
        P
            Pressure in LAMMPS pressure units for NPT, or `None` for NVT.
        seed
            Random seed for velocity generation and passed to `setup`.
        potential
            External potential calculator derived from `mdext.potential.Potential`.
        units
            Supported LAMMPS units system (see `unit_names`).
        timestep
            MD timestep in LAMMPS time units.
        n_thermo
            Number of time steps between each thermo and data collection call.
        thermo_per_cycle
            Number of thermo and data collection calls per reporting cycle.
            Averaged thermo output is reported at this interval,
            and if requested, densities are updated to file at this interval.
        dr
            Spatial resolution for density collection in LAMMPS distance units.
            Note that densities are collected on a 1D planar, cylindrical or spherical
            grid based on whether `potential` is planar, cylindrical or spherical.
        Tdamp
            Thermostat damping time in LAMMPS time units for NVT / NPT.
        Pdamp
            Barostat damping time in LAMMPS time units (used only for NPT).
        """

        # Set up LAMMPS instance:
        self.is_head = (MPI.COMM_WORLD.rank == 0)
        lps = lammps()
        os.environ["OMP_NUM_THREADS"] = "1"  # run single-threaded
        lmp = PyLammps(ptr=lps)
        self.lmp = lmp
        self.t_start = time.time()

        # Global settings:
        assert units in unit_names
        self.units = units
        self.unit_names = unit_names[units]
        lmp.units(units)
        lmp.dimension("3")
        lmp.boundary("p p p")
        
        # Set up initial atomic configuration and interaction potential:
        setup(lmp, seed)

        # Prepare for dynamics:
        lmp.reset_timestep(0)
        lmp.timestep(timestep)  # in femtoseconds
        lmp.velocity(f"all create {T} {seed} dist gaussian loop local")
        self.T = T
        self.P = P
        if P is None:
            lmp.fix(f"Ensemble all nvt temp {T} {T} {Tdamp}")
        else:
            lmp.fix(f"Ensemble all npt temp {T} {T} {Tdamp} iso {P} {P} {Pdamp}")

        # Setup thermo callback
        lmp.thermo(n_thermo)
        mdext.md.thermo_callback = self
        lmp.python(
            "thermo_callback input 1 SELF return v_thermo_callback format pf here"
            " 'from mdext.md import thermo_callback'"
        )
        lmp.variable("thermo_callback python thermo_callback")
        lmp.thermo_style("custom step temp press pe vol v_thermo_callback")
        self.n_thermo = n_thermo
        self.thermo_per_cycle = thermo_per_cycle
        self.steps_per_cycle = n_thermo * thermo_per_cycle
        log.info(
            f"Time[{self.unit_names['time']}] per step: {timestep}"
            f"  thermo: {self.thermo_per_cycle * timestep}"
            f"  cycle: {self.steps_per_cycle * timestep}"
        )
        
        # Set up external force callback
        self.potential = potential
        lmp.fix("ext all external pf/callback 1 1")
        lps.set_fix_external_callback("ext", self.potential, lps)
        
        # Prepare for data collection:
        self.i_thermo = -1  # index of current thermo entry within cycle
        self.i_cycle = 0  # index of current cycle
        self.cycle_stats = np.zeros(4)  # cumulative T, P, PE, vol
        # --- setup histograms:
        self.dr = dr
        boxlo, boxhi = lps.extract_box()[:2]
        self.hist = Histogram(boxlo[2], boxhi[2], self.dr, 1)  # used in each thermo
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

    def run(self, n_cycles: int, run_name: str, out_filename: str = "") -> None:
        """
        Run `n_cycles` cycles of `thermo_per_cycle` x `n_thermo` steps each.
        Use `run_name` to report the start and end of the run (to ease parsing the log).
        If `out_filename` is specified, save density response (HDF5) after every cycle.
        """
        log.info(f"Starting {run_name}")
        log.info(
            f"Cycle T[{self.unit_names['temperature']}]"
            f" P[{self.unit_names['pressure']}] PE[{self.unit_names['energy']}]"
            f" vol[{self.unit_names['distance']}^3] t_cpu[s]"
        )
        for i_cycle in range(n_cycles):
            self.lmp.run(self.steps_per_cycle)
            if out_filename:
                self.save_response(out_filename)
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

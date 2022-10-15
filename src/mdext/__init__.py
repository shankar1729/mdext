"""Molecular dynamics in external potentials using PyLAMMPS"""
# List exported symbols for doc generation
__all__ = [
    "MPI",
    "md",
    "potential",
    "make_liquid",
    "log",
]

# Module import definition
from mpi4py import MPI
from . import md, potential, histogram, make_liquid
from .md import log

# Automatic versioning added by versioneer
from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions

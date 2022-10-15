import numpy as np
from scipy.spatial.transform import Rotation
from typing import List


def make_water(
    *,
    pos_min: List[float],
    pos_max: List[float],
    out_file: str,
    N_bulk: float = 0.033325,
    r_OH: float = 1.00,
    theta_HOH: float = 109.47,
    max_attempts: int = 1000,
) -> None:
    """Make a box of water molecules using `make_liquid`.
    Box size and output parameters are same as in `make_liquid`,
    while molecule geometry parameters are constructed based on
    bond length `r_OH` (Angstroms) and angle `theta_HOH` (degrees).
    Defaults to SPC/E molecular geometry.
    """
    hlf_theta_rad = np.deg2rad(0.5 * theta_HOH)
    cos, sin = np.cos(hlf_theta_rad), np.sin(hlf_theta_rad)
    make_liquid(
        pos_min=pos_min,
        pos_max=pos_max,
        out_file=out_file,
        N_bulk=N_bulk,
        masses=[1.0079401, 15.999400 ],
        radii=[0.5, 0.9],
        atom_types=[2, 1, 1],
        atom_pos=[
            [0., 0., 0.],
            [0., r_OH * cos, +r_OH * sin],
            [0., r_OH * cos, -r_OH * sin],
        ],
        bond_types=[1, 1],
        bond_indices=[[0, 1], [0, 2]],
        angle_types=[1],
        angle_indices=[[1, 0, 2]],
        max_attempts=max_attempts,
    )


def make_liquid(
    *,
    pos_min: List[float],
    pos_max: List[float],
    out_file: str,
    N_bulk: float,
    masses: List[float],
    radii: List[float],
    atom_types: List[int],
    atom_pos: List[List[float]],
    bond_types: List[int],
    bond_indices: List[List[int]],
    angle_types: List[int],
    angle_indices: List[List[int]],
    max_attempts: int = 1000,
) -> None:
    """
    Create a box filled with a liquid with randomly-oriented molecules
    and write to LAMMPS data file, given molecule geometry and liquid
    parameters. See `make_water` for an example of customizing this
    for a specific molecular liquid.
    
    Parameters
    ----------
    pos_min
        Lowest coordinate in each direction of bounding box of liquid.
    pos_max
        Highest coordinate in each direction of bounding box of liquid.
    out_file
        Output file name (LAMMPS data format).
    N_bulk
        Bulk number density of molecules in liquid (in Ansgtrom^-3)
    masses
        Mass of each atom type (one for each LAMMPS atom type, in amu).
    radii
        Corresponding radii (in Angstroms) to test intersection while placing molecules
        within the box. Recommend using values between the covalent and vdW radii.
    atom_types
        Atom types within molecule (1-based LAMMPS atom types).    
    atom_pos
        Positions (in Angstroms) of each atom within molecule in reference orientation.
    bond_types
        Bond types within molecule (1-based LAMMPS bond types).
    bond_indices
        Lists of pairs of atom indices connected by each bond within one molecule
        (effectively 0-based indices to the `atom_pos` array).
    angle_types
        Angle types within molecule (1-based LAMMPS angle types).
    angle_indices
        Lists of triplets of atom indices connected by each angle within one molecule
        (effectively 0-based indices to the `atom_pos` array).
    max_attempts
        Maximum attempts at inserting any given molecule."""

    # Overall geometry parameters:
    pos_min = np.array(pos_min)
    pos_max = np.array(pos_max)
    L = pos_max - pos_min
    assert np.all(L > 0.)
    n_mols = int(L.prod() * N_bulk)

    # Individual molecule geometry:
    radii = np.array(radii)
    atom_pos = np.array(atom_pos)
    atom_types = np.array(atom_types, dtype=int)
    atom_radii = radii[atom_types - 1]  # radius of each atom in molecule
    bond_types = np.array(bond_types, dtype=int)
    bond_indices = np.array(bond_indices, dtype=int)
    angle_types = np.array(angle_types, dtype=int)
    angle_indices = np.array(angle_indices, dtype=int)

    # Initialize global arrays:
    # --- atoms
    n_atoms = n_mols * len(atom_pos)  # total number of atoms
    n_atoms_done = 0
    atom_pos_all = np.zeros((n_atoms, 3))
    atom_types_all = np.zeros(n_atoms, dtype=int)
    atom_radii_all = np.zeros(n_atoms)
    molecule_ids = np.zeros(n_atoms, dtype=int)
    # --- bonds
    n_bonds = n_mols * len(bond_types)
    n_bonds_done = 0
    bond_types_all = np.zeros(n_bonds, dtype=int)
    bond_indices_all = np.zeros((n_bonds, 2), dtype=int)
    # --- angles
    n_angles = n_mols * len(angle_types)
    n_angles_done = 0
    angle_types_all = np.zeros(n_angles, dtype=int)
    angle_indices_all = np.zeros((n_angles, 3), dtype=int)

    def try_insert_molecule(molecule_id: int) -> bool:
        """Attempt to insert a molecule and return whether successful.
        Here, `molecule_id` is the 1-based index of inserted molecule."""

        # Generate candidate atomic positions:
        pos0 = pos_min + L * np.random.rand(3)  # random translation
        rot = Rotation.random()  # random rotation
        pos = pos0 + rot.apply(atom_pos)

        # Check bounding box:
        if np.any(pos < pos_min[None, :] + atom_radii[:, None]):
            return False
        if np.any(pos > pos_max[None, :] - atom_radii[:, None]):
            return False

        # Check previous atoms:
        nonlocal n_atoms_done, n_bonds_done, n_angles_done
        dist_sq = (
            (pos[None, ...] - atom_pos_all[:n_atoms_done, None, :]) ** 2
        ).sum(axis=-1)
        if np.any(
            dist_sq < (atom_radii[None, :] + atom_radii_all[:n_atoms_done, None]) ** 2
        ):
            return False
        
        # Add molecule:
        # --- atoms
        n_atoms_next = n_atoms_done + len(atom_pos)
        atom_slice = slice(n_atoms_done, n_atoms_next)
        atom_pos_all[atom_slice] = pos
        atom_types_all[atom_slice] = atom_types
        atom_radii_all[atom_slice] = atom_radii
        molecule_ids[atom_slice] = molecule_id
        # --- bonds
        n_bonds_next = n_bonds_done + len(bond_types)
        bond_slice = slice(n_bonds_done, n_bonds_next)
        bond_types_all[bond_slice] = bond_types
        bond_indices_all[bond_slice] = bond_indices + n_atoms_done + 1  # 1-based
        # --- angles
        n_angles_next = n_angles_done + len(angle_types)
        angle_slice = slice(n_angles_done, n_angles_next)
        angle_types_all[angle_slice] = angle_types
        angle_indices_all[angle_slice] = angle_indices + n_atoms_done + 1  # 1-based
        # --- update counts
        n_atoms_done = n_atoms_next
        n_bonds_done = n_bonds_next
        n_angles_done = n_angles_next
        return True

    def insert_molecule(molecule_id: int) -> int:
        """Insert molecule and return number of attempts.
        Raise exception if max_attempts exceeded"""
        for i_attempt in range(max_attempts):
            if try_insert_molecule(molecule_id):
                return i_attempt + 1
        raise IndexError(f"Exceeded {max_attempts} attempts")

    # Insert molecules sequentially:
    n_attempts_tot = 0
    n_attempts_max = 0
    for molecule_id in range(1, n_mols + 1):
        n_attempts = insert_molecule(molecule_id)
        n_attempts_tot += n_attempts
        n_attempts_max = max(n_attempts_max, n_attempts)
    assert n_atoms_done == n_atoms
    assert n_bonds_done == n_bonds
    assert n_angles_done == n_angles
    print(
        f"Inserted {n_mols} molecules with {n_attempts_tot / n_mols:.1f}"
        f" average and {n_attempts_max} maximum attempts."
    )
    
    # Write LAMMPS data file:
    with open(out_file, "w") as fp:
        fp.write("Random liquid created from single molecule by make_liquid\n")
        fp.write(f"{atom_types_all.max():4d} atom types\n")
        fp.write(f"{bond_types_all.max():4d} bond types\n")
        fp.write(f"{angle_types_all.max():4d} angle types\n")
        fp.write(f"{n_atoms:4d} atoms\n")
        fp.write(f"{n_bonds:4d} bonds\n")
        fp.write(f"{n_angles:4d} angles\n")
        for dir_min, dir_max, dir_name in zip(pos_min, pos_max, "xyz"):
            fp.write(f" {dir_min} {dir_max} {dir_name}lo {dir_name}hi\n")

        fp.write("\nMasses\n\n")
        for i_mass, mass in enumerate(masses):
            fp.write(f"{i_mass + 1} {mass}\n")
        
        fp.write("\nAtoms\n\n")
        for i_atom, (molecule_id, atom_type, (x, y, z)) in enumerate(
            zip(molecule_ids, atom_types_all, atom_pos_all)
        ):
            fp.write(f"{i_atom + 1} {molecule_id} {atom_type} 0 {x} {y} {z}\n")

        fp.write("\nBonds\n\n")
        for i_bond, (bond_type, (at1, at2)) in enumerate(
            zip(bond_types_all, bond_indices_all)
        ):
            fp.write(f"{i_bond + 1} {bond_type} {at1} {at2}\n")

        fp.write("\nAngles\n\n")
        for i_angle, (angle_type, (at1, at2, at3)) in enumerate(
            zip(angle_types_all, angle_indices_all)
        ):
            fp.write(f"{i_angle + 1} {angle_type} {at1} {at2} {at3}\n")


if __name__ == "__main__":
    make_water(pos_min=[-10, -10, -21.5], pos_max=[10, 10, 21.5], out_file="test.dat")

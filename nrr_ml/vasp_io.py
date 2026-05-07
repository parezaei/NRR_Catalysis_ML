"""
vasp_io.py
==========
Low-level I/O utilities for VASP output files.

Functions
---------
get_electronegativity        : Pauling electronegativity for an element symbol.
get_valence_electrons        : Parse ZVAL per element from a POTCAR file.
read_CONTCAR                 : Parse a CONTCAR/POSCAR into a list or DataFrame.
bader_df                     : Parse an ACF.dat Bader charge file into a DataFrame.
extract_total_energy         : Extract the final TOTEN from an OUTCAR file.
is_converged                 : Check whether a VASP geometry optimisation converged.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Potcar


# ---------------------------------------------------------------------------
# Elemental properties
# ---------------------------------------------------------------------------

def get_electronegativity(atom_type: str) -> float:
    """Return the Pauling electronegativity of *atom_type* (e.g. ``'Fe'``).

    Parameters
    ----------
    atom_type:
        Element symbol recognised by pymatgen.

    Returns
    -------
    float
        Pauling electronegativity.
    """
    return Element(atom_type).X


def get_valence_electrons(potcar_file: str) -> dict[str, float]:
    """Parse a POTCAR file and return the number of valence electrons per element.

    Parameters
    ----------
    potcar_file:
        Absolute path to the POTCAR file.

    Returns
    -------
    dict
        ``{element_symbol: n_valence_electrons}``.

    Raises
    ------
    FileNotFoundError
        If *potcar_file* does not exist.
    RuntimeError
        On any other pymatgen parsing error.
    """
    try:
        potcar = Potcar.from_file(potcar_file)
        return {entry.element: entry.nelectrons for entry in potcar}
    except FileNotFoundError:
        raise FileNotFoundError(f"POTCAR not found: {potcar_file}")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse POTCAR: {exc}") from exc


# ---------------------------------------------------------------------------
# Structure file parsing
# ---------------------------------------------------------------------------

def read_CONTCAR(
    file_path: str,
    fractional: bool = False,
    output_type: str = "df",
):
    """Parse a CONTCAR or POSCAR file.

    Parameters
    ----------
    file_path:
        Path to the CONTCAR/POSCAR file.
    fractional:
        If ``True``, return fractional coordinates; otherwise Cartesian (Å).
    output_type:
        ``'df'`` returns a :class:`pandas.DataFrame`; ``'list'`` returns a
        list of ``(element, coordinate_array)`` tuples.

    Returns
    -------
    pandas.DataFrame or list
        Atomic coordinates with columns ``['element', 'x', 'y', 'z']``
        (DataFrame mode) or ``[(element, np.ndarray), ...]`` (list mode).
    """
    structure = Structure.from_file(file_path)

    records = []
    for site in structure:
        coords = site.frac_coords if fractional else site.coords
        records.append({
            "element": site.specie.symbol,
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
        })

    if output_type == "df":
        return pd.DataFrame(records)
    return [(r["element"], np.array([r["x"], r["y"], r["z"]])) for r in records]


# ---------------------------------------------------------------------------
# Bader charge parsing
# ---------------------------------------------------------------------------

#: Default valence electrons — override by passing *valence_dict* explicitly.
_DEFAULT_VALENCE: dict[str, float] = {
    "Ag": 19.0, "Al": 3.0,  "Au": 11.0, "Co": 9.0,
    "Cu": 19.0, "Fe": 16.0, "H":  1.0,  "N":  5.0,
    "Ni": 10.0, "Pd": 16.0, "Pt": 16.0, "Ru": 16.0,
    "V":  13.0, "Zn": 12.0,
}


def bader_df(
    acf_dat: str,
    input_path: str,
    input_type: str = "vasp",
    valence_dict: dict[str, float] | None = None,
    slab: bool = True,
    atom_types_line_num: int = 5,
    type_num_line_num: int = 6,
    z_threshold: float = 1.0,
    molecule_exists: bool = False,
    num_atom_mol: int = 2,
) -> pd.DataFrame:
    """Parse a Bader ACF.dat file and compute net Bader charges.

    Supports both VASP (POSCAR) and Quantum ESPRESSO (``input.in``) input
    formats. The resulting DataFrame contains one row per atom with columns:

    ``atom_type``, ``x``, ``y``, ``z``, ``Charge``, ``Min_dist``,
    ``atomic_volume``, ``bader_charge`` (= valence − Bader charge),
    and — when *slab* is ``True`` — ``surface_atom`` (1 if surface layer).

    Parameters
    ----------
    acf_dat:
        Path to the ``ACF.dat`` file produced by the Bader code.
    input_path:
        Path to the corresponding POSCAR (VASP) or ``input.in`` (QE) file.
    input_type:
        ``'vasp'`` or ``'qs'``.
    valence_dict:
        Mapping ``{element: n_valence}``. Defaults to :data:`_DEFAULT_VALENCE`.
    slab:
        If ``True``, flag atoms in the topmost layer as surface atoms.
    atom_types_line_num:
        Line index (0-based) in POSCAR listing element symbols (VASP only).
    type_num_line_num:
        Line index (0-based) in POSCAR listing atom counts (VASP only).
    z_threshold:
        Å tolerance used to identify the surface layer (VASP/QE).
    molecule_exists:
        Set ``True`` when the POSCAR includes adsorbate atoms.
    num_atom_mol:
        Number of adsorbate atoms at the top of the structure.

    Returns
    -------
    pandas.DataFrame
    """
    if valence_dict is None:
        valence_dict = _DEFAULT_VALENCE

    # --- Read ACF.dat ---
    with open(acf_dat) as fh:
        acf_lines = fh.readlines()[2:-4]

    col_names = ["#", "x", "y", "z", "Charge", "Min_dist", "atomic_volume"]
    acf = pd.DataFrame()
    for i, col in enumerate(col_names):
        acf[col] = [
            float(re.findall(r"-*\d+\.*\d*", line.split()[i])[0])
            for line in acf_lines
        ]
    acf.drop("#", axis=1, inplace=True)

    # --- Read structure file ---
    with open(input_path) as fh:
        lines = fh.readlines()

    z_values = list(acf["z"])

    if input_type == "qs":
        nat = next(
            int(re.findall(r"\d+", l)[0])
            for l in lines if "nat" in l
        )
        start = next(i + 1 for i, l in enumerate(lines) if "ATOMIC_POSITIONS" in l)
        atoms_list = [lines[start + i].split()[0] for i in range(nat)]

    elif input_type == "vasp":
        atom_types = lines[atom_types_line_num].split()
        type_counts = [int(n) for n in lines[type_num_line_num].split()]
        atoms_list = [
            atype
            for atype, count in zip(atom_types, type_counts)
            for _ in range(count)
        ]
    else:
        raise ValueError(f"input_type must be 'vasp' or 'qs', got '{input_type}'")

    z_max = sorted(z_values, reverse=True)[num_atom_mol if molecule_exists else 0]

    acf["z_alat"] = z_values
    acf["atom_type"] = atoms_list
    acf = acf[["atom_type", "x", "y", "z", "Charge", "Min_dist", "atomic_volume", "z_alat"]]

    if slab:
        acf["surface_atom"] = (acf["z_alat"] > z_max - z_threshold).astype(int)

    acf.drop("z_alat", axis=1, inplace=True)
    valence_arr = np.array([valence_dict.get(a, 0.0) for a in atoms_list])
    acf["bader_charge"] = valence_arr - acf["Charge"].values

    return acf


# ---------------------------------------------------------------------------
# OUTCAR utilities
# ---------------------------------------------------------------------------

def extract_total_energy(outcar_file: str) -> float | None:
    """Return the last ``free  energy   TOTEN`` value from an OUTCAR file.

    Parameters
    ----------
    outcar_file:
        Path to the VASP OUTCAR file.

    Returns
    -------
    float or None
        Total energy in eV, or ``None`` if not found.
    """
    last_energy = None
    try:
        with open(outcar_file) as fh:
            for line in fh:
                if "free  energy   TOTEN" in line:
                    last_energy = float(line.split()[-2])
    except FileNotFoundError:
        raise FileNotFoundError(f"OUTCAR not found: {outcar_file}")
    return last_energy


def is_converged(outcar_file: str) -> bool:
    """Return ``True`` if a VASP geometry optimisation converged.

    Checks for the string
    ``'reached required accuracy - stopping structural energy minimisation'``
    in the OUTCAR.

    Parameters
    ----------
    outcar_file:
        Path to the VASP OUTCAR file.

    Returns
    -------
    bool
    """
    with open(outcar_file) as fh:
        return any(
            "reached required accuracy - stopping structural energy minimisation" in line
            for line in fh
        )

"""
structure.py
============
Structure manipulation and neighbour-finding utilities for VASP slab calculations.

Functions
---------
expand_struct_vasp                         : Create a supercell POSCAR.
expand_charge_file_vasp                    : Expand a Bader ACF.dat to match a supercell.
calculate_nearest_slab_neighbors           : Find N nearest slab atoms to the adsorbate centre.
get_neighbors_with_bader_and_electronegativity : Full neighbour descriptor including Bader charge.
calculate_molecule_distances               : Bond lengths within an adsorbate from a CONTCAR.
extract_molecule_coords                    : Pull adsorbate atom coordinates from a results dict.
mol_surf_distance_dict                     : Organise mol–metal distances by atom label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from .vasp_io import (
    get_electronegativity,
    get_valence_electrons,
    read_CONTCAR,
    bader_df,
    _DEFAULT_VALENCE,
)


# ---------------------------------------------------------------------------
# Supercell utilities
# ---------------------------------------------------------------------------

def expand_struct_vasp(
    input_path: str,
    output_path: str,
    supercell_size: list[int],
) -> None:
    """Create a supercell POSCAR from an existing POSCAR/CONTCAR.

    Parameters
    ----------
    input_path:
        Path to the original POSCAR.
    output_path:
        Destination path for the expanded POSCAR.
    supercell_size:
        Three-element list ``[na, nb, nc]`` defining the supercell dimensions.
    """
    structure = Structure.from_file(input_path)
    structure.make_supercell(supercell_size)
    structure.to(fmt="POSCAR", filename=output_path)


def expand_charge_file_vasp(
    acf_dat: str,
    input_path: str,
    potcar_path: str,
    input_type: str = "vasp",
    supercell_size: list[int] | None = None,
) -> pd.DataFrame:
    """Expand a Bader ACF.dat file to match a supercell.

    Takes an unexpanded ACF.dat and POSCAR pair, creates the supercell POSCAR,
    tile the charge entries, recomputes Bader charges, and returns the full
    expanded Bader DataFrame with Cartesian coordinates.

    Parameters
    ----------
    acf_dat:
        Path to the original (unexpanded) ACF.dat file.
    input_path:
        Path to the original (unexpanded) POSCAR file.
    potcar_path:
        Path to the POTCAR file (used to obtain valence electrons).
    input_type:
        ``'vasp'`` (default) or ``'qs'``.
    supercell_size:
        ``[na, nb, nc]`` — defaults to ``[4, 4, 1]``.

    Returns
    -------
    pandas.DataFrame
        Expanded Bader DataFrame aligned to the supercell geometry.
    """
    if supercell_size is None:
        supercell_size = [4, 4, 1]

    valence_electrons = get_valence_electrons(potcar_path)
    repetition = int(np.prod(supercell_size))
    size_str = "".join(str(s) for s in supercell_size)

    new_poscar_path = f"{input_path}.{size_str}_dummy"
    new_acf_path = f"{acf_dat}.{size_str}"
    new_acf_dummy_path = f"{acf_dat}.dummy_{size_str}"

    # Build supercell POSCAR
    structure = Structure.from_file(input_path)
    structure.make_supercell(supercell_size)
    structure.to(fmt="POSCAR", filename=new_poscar_path)

    # Tile the ACF.dat entries
    _NON_REPEAT = ["---", "X", "VACUUM", "NUMBER"]
    with open(acf_dat) as fh:
        acf_lines = fh.readlines()

    with open(new_acf_dummy_path, "w") as fh:
        for line in acf_lines:
            if any(tag in line for tag in _NON_REPEAT):
                fh.write(line)
            else:
                fh.write(line * repetition)

    # Parse the tiled file using the supercell POSCAR
    expanded = bader_df(
        acf_dat=new_acf_dummy_path,
        input_type="vasp",
        input_path=new_poscar_path,
        valence_dict=valence_electrons,
        slab=True,
        atom_types_line_num=5,
        type_num_line_num=6,
        z_threshold=1.0,
    )

    # Overwrite coordinates from the actual supercell POSCAR
    poscar_df = read_CONTCAR(new_poscar_path, output_type="df")
    expanded["x"] = poscar_df["x"].values
    expanded["y"] = poscar_df["y"].values
    expanded["z"] = poscar_df["z"].values

    return expanded


# ---------------------------------------------------------------------------
# Neighbour finding
# ---------------------------------------------------------------------------

def calculate_nearest_slab_neighbors(
    poscar_file: str,
    molecule_atom_types: list[str],
    num_neighbor: int = 20,
) -> dict:
    """Find the *num_neighbor* nearest slab atoms to the adsorbate centre.

    The adsorbate centre is defined as:

    * midpoint of the two N atoms (N₂ or N₂H),
    * single N atom coordinate (one N atom),
    * midpoint of any two adsorbate atoms (biatomic without N),
    * centroid of all adsorbate atoms (otherwise).

    Parameters
    ----------
    poscar_file:
        Path to a POSCAR containing both slab and adsorbate atoms.
    molecule_atom_types:
        Element symbols that belong to the adsorbate (e.g. ``['N', 'H']``).
    num_neighbor:
        Number of nearest slab atoms to return.

    Returns
    -------
    dict
        ``{'molecule_center': list[float],
           'nearest_atoms': list[tuple(type, coords, dist_vec, dist, index)]}``
    """
    with open(poscar_file) as fh:
        content = fh.read()

    structure = Poscar.from_string(content).structure
    lattice = structure.lattice.matrix
    all_coords = np.dot(structure.frac_coords, lattice)
    all_types = [str(sp) for sp in structure.species]

    mol_indices = [
        i for i, t in enumerate(all_types) if t in molecule_atom_types
    ]
    if not mol_indices:
        raise ValueError(
            f"No atoms of types {molecule_atom_types} found in {poscar_file}."
        )

    mol_coords = all_coords[mol_indices]
    n_coords = all_coords[[i for i, t in enumerate(all_types) if t == "N"]]

    if len(n_coords) == 2:
        centre = n_coords.mean(axis=0)
    elif len(n_coords) == 1:
        centre = n_coords[0]
    elif len(mol_coords) == 2:
        centre = mol_coords.mean(axis=0)
    else:
        centre = mol_coords.mean(axis=0)

    slab_info = []
    for i, (coord, atype) in enumerate(zip(all_coords, all_types)):
        if i in mol_indices:
            continue
        vec = coord - centre
        dist = float(np.linalg.norm(vec))
        slab_info.append((atype, tuple(coord), tuple(vec), dist, i))

    slab_info.sort(key=lambda x: x[3])

    return {
        "molecule_center": centre.tolist(),
        "nearest_atoms": slab_info[:num_neighbor],
    }


def get_neighbors_with_bader_and_electronegativity(
    poscar_file: str,
    molecule_atom_types: list[str],
    acf_surf: str,
    poscar_surf: str,
    potcar_surf: str,
    supercell_size: list[int] | None = None,
    num_neighbor: int = 20,
) -> dict:
    """Extend nearest-neighbour data with Bader charges and electronegativities.

    Combines :func:`calculate_nearest_slab_neighbors` with Bader charge
    data from the expanded surface ACF.dat.

    Parameters
    ----------
    poscar_file:
        POSCAR containing slab + adsorbate (post-relaxation).
    molecule_atom_types:
        Adsorbate element symbols.
    acf_surf:
        Path to the unexpanded surface ACF.dat.
    poscar_surf:
        Path to the unexpanded surface POSCAR.
    potcar_surf:
        Path to the surface POTCAR.
    supercell_size:
        Supercell dimensions used during the DFT calculation (default ``[4,4,1]``).
    num_neighbor:
        Number of nearest slab atoms.

    Returns
    -------
    dict
        ``{'molecule_center': ...,
           'nearest_atoms': [{'atom_type', 'coordinates', 'distance_vector',
                              'distance', 'index', 'bader_charge',
                              'electronegativity'}, ...]}``
    """
    if supercell_size is None:
        supercell_size = [4, 4, 1]

    nn_data = calculate_nearest_slab_neighbors(
        poscar_file, molecule_atom_types, num_neighbor
    )
    bader = expand_charge_file_vasp(
        acf_surf, poscar_surf, potcar_surf, supercell_size=supercell_size
    )

    enriched = []
    for (atype, cart, vec, dist, idx) in nn_data["nearest_atoms"]:
        try:
            charge = float(bader.loc[idx, "bader_charge"])
        except KeyError:
            charge = None

        try:
            elneg = get_electronegativity(atype)
        except Exception:
            elneg = None

        enriched.append({
            "atom_type": atype,
            "coordinates": cart,
            "distance_vector": vec,
            "distance": dist,
            "index": idx,
            "bader_charge": charge,
            "electronegativity": elneg,
        })

    return {"molecule_center": nn_data["molecule_center"], "nearest_atoms": enriched}


# ---------------------------------------------------------------------------
# Adsorbate geometry utilities
# ---------------------------------------------------------------------------

def calculate_molecule_distances(
    poscar_file: str,
    molecule_atoms: list[str],
) -> dict:
    """Return pairwise bond lengths between adsorbate atoms in a CONTCAR.

    Accounts for periodic boundary conditions via pymatgen.

    Parameters
    ----------
    poscar_file:
        Path to the CONTCAR/POSCAR containing the adsorbate.
    molecule_atoms:
        Element symbols that belong to the adsorbate (e.g. ``['N', 'H']``).

    Returns
    -------
    dict
        ``{'bond_label': distance_in_Angstrom, ...}`` — e.g.
        ``{'N-N': 1.10, 'N-H1': 1.02, 'N-H2': 1.02}``.
    """
    structure = Structure.from_file(poscar_file)
    mol_sites = [s for s in structure if s.specie.symbol in molecule_atoms]

    distances = {}
    for i, si in enumerate(mol_sites):
        for j, sj in enumerate(mol_sites):
            if j <= i:
                continue
            label = f"{si.specie.symbol}-{sj.specie.symbol}{j}"
            distances[label] = float(structure.get_distance(
                structure.index(si), structure.index(sj)
            ))
    return distances


def extract_molecule_coords(results: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Extract adsorbate atom types and Cartesian coordinates from a results dict.

    Parameters
    ----------
    results:
        List of atom dictionaries as produced by neighbour-finding functions,
        each containing ``'atom_type'`` and ``'coordinates'``.

    Returns
    -------
    list of (atom_type, coordinate_array) tuples.
    """
    return [
        (atom["atom_type"], np.array(atom["coordinates"]))
        for atom in results
        if atom["atom_type"] in {"N", "H"}
    ]


def mol_surf_distance_dict(mol_surf_results: list[dict]) -> dict:
    """Organise mol–metal distances by atom label (N0, N1, H0, …).

    Parameters
    ----------
    mol_surf_results:
        Output list from a nearest-neighbour search that includes adsorbate
        atoms (with ``'molecule_atom'`` and ``'distance'`` keys).

    Returns
    -------
    dict
        ``{'N0-metal': float | None, 'N1-metal': float | None,
           'H0-metal': float | None, 'H1-metal': float | None,
           'H2-metal': float | None}``
    """
    result = {k: None for k in ("N0-metal", "N1-metal", "H0-metal", "H1-metal", "H2-metal")}
    n_count = h_count = 0

    for atom in mol_surf_results:
        atype = atom.get("molecule_atom", "")
        dist = atom.get("distance", None)

        if atype == "N" and n_count < 2:
            result[f"N{n_count}-metal"] = dist
            n_count += 1
        elif atype == "H" and h_count < 3:
            result[f"H{h_count}-metal"] = dist
            h_count += 1

    return result

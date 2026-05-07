"""
features.py
===========
DFT-derived feature extraction: projected density of states (PDOS) band
centres / filling fractions per neighbour shell.

Functions
---------
calculate_orbital_band_info : Compute s/p/d band centres and filling fractions
                               for a set of atom indices from a vasprun.xml.
add_dos_features            : Apply *calculate_orbital_band_info* over an
                               entire DataFrame, adding ``n0_n{k}_dos`` columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.integrate import trapezoid

from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.io.vasp import Vasprun


# ---------------------------------------------------------------------------
# Band-centre / filling computation
# ---------------------------------------------------------------------------

def calculate_orbital_band_info(
    vasprun_path: str,
    target_indices: list[int] | None = None,
    emin: float = -10.0,
    emax: float = 10.0,
) -> dict:
    """Compute s, p, and d band centres and filling fractions.

    For a set of atom indices in a VASP calculation, sum the projected DOS
    over those atoms and compute the first moment (band centre, ε) and
    the zeroth moment (integrated DOS, filling fraction f) in the energy
    window [*emin*, *emax*] relative to the Fermi level.

    Two projections are reported for each orbital:

    * ``selected`` — summed over *target_indices* only.
    * ``total``    — summed over all atoms in the structure.

    Parameters
    ----------
    vasprun_path:
        Path to a ``vasprun.xml`` (or ``vasprun.xml.gz``) file.
    target_indices:
        0-based pymatgen site indices to project onto (the "local" shell).
        If ``None`` or empty, only the total projection is returned.
    emin:
        Lower bound of the energy integration window (eV, relative to E_F).
    emax:
        Upper bound of the energy integration window (eV, relative to E_F).

    Returns
    -------
    dict
        Flat dictionary with keys of the form
        ``'{selected|total}_{s|p|d}band_{center|sum}'``.
        Returns ``{'error': str}`` on any exception.
    """
    try:
        vr = Vasprun(vasprun_path, parse_dos=True, parse_eigen=False,
                     parse_projected_eigen=False)
        cdos = vr.complete_dos
        energies = cdos.energies - cdos.efermi
        structure = cdos.structure

        mask = (energies >= emin) & (energies <= emax)
        E = energies[mask]

        orbitals = {"s": OrbitalType.s, "p": OrbitalType.p, "d": OrbitalType.d}

        def _to_array(dos_obj) -> np.ndarray:
            dens = dos_obj.densities
            if Spin.up in dens and Spin.down in dens:
                return dens[Spin.up] + dens[Spin.down]
            return list(dens.values())[0]

        def _band_stats(dos_full: np.ndarray) -> tuple[float | None, float]:
            dos_win = dos_full[mask]
            band_sum = float(trapezoid(dos_win, E))
            if band_sum != 0:
                band_center = float(trapezoid(dos_win * E, E) / band_sum)
            else:
                band_center = None
            return band_center, band_sum

        results: dict = {}

        # Total projection (all atoms)
        for orb_label, orb_type in orbitals.items():
            total = np.zeros_like(energies)
            for site in structure:
                spd = cdos.get_site_spd_dos(site)
                if orb_type in spd:
                    total += _to_array(spd[orb_type])
            center, filling = _band_stats(total)
            results[f"total_{orb_label}band_center"] = center
            results[f"total_{orb_label}band_sum"] = filling

        # Selected projection (target_indices only)
        if target_indices:
            for orb_label, orb_type in orbitals.items():
                selected = np.zeros_like(energies)
                for idx in target_indices:
                    if idx >= len(structure):
                        continue
                    spd = cdos.get_site_spd_dos(structure[idx])
                    if orb_type in spd:
                        selected += _to_array(spd[orb_type])
                center, filling = _band_stats(selected)
                results[f"selected_{orb_label}band_center"] = center
                results[f"selected_{orb_label}band_sum"] = filling
            results["selected_indices"] = target_indices

        return results

    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# DataFrame-level feature addition
# ---------------------------------------------------------------------------

def add_dos_features(
    df: pd.DataFrame,
    base_vasprun_dir: str,
    max_neighbors: int = 20,
    emin: float = -10.0,
    emax: float = 10.0,
    vasprun_subpath: str = "1_surf/postprocess/2_DOS/vasprun.xml",
    person_col: str = "person",
    material_col: str = "material",
) -> pd.DataFrame:
    """Add cumulative DOS feature columns ``n0_n{k}_dos`` to a DataFrame.

    For each row and each neighbour level *k* from 0 to *max_neighbors*, the
    function computes :func:`calculate_orbital_band_info` using the indices of
    atoms ``n0 … nk`` as the ``target_indices``.

    Parameters
    ----------
    df:
        Input DataFrame containing neighbour index columns ``n0_index``,
        ``n1_index``, …, ``n{max_neighbors}_index``, as well as *person_col*
        and *material_col* to resolve vasprun paths.
    base_vasprun_dir:
        Root directory under which per-person / per-material subfolders live.
    max_neighbors:
        Maximum neighbour shell index (inclusive).
    emin, emax:
        DOS integration window (eV, relative to E_F).
    vasprun_subpath:
        Path appended to ``base_vasprun_dir / person / material`` to find the
        vasprun.xml (e.g. ``'1_surf/postprocess/2_DOS/vasprun.xml'``).
    person_col, material_col:
        Column names identifying the dataset contributor and alloy label.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with additional columns ``n0_n0_dos`` … ``n0_n{max_neighbors}_dos``.
    """
    df = df.copy()

    for k in range(max_neighbors + 1):
        col_name = f"n0_n{k}_dos"
        index_cols = [f"n{j}_index" for j in range(k + 1)]

        def _get_dos(row, index_cols=index_cols):
            try:
                indices = [
                    int(row[c]) for c in index_cols if pd.notna(row.get(c))
                ]
                vasprun_path = (
                    Path(base_vasprun_dir)
                    / row[person_col]
                    / row[material_col]
                    / vasprun_subpath
                )
                return calculate_orbital_band_info(
                    str(vasprun_path), indices, emin=emin, emax=emax
                )
            except Exception as exc:
                return {"error": str(exc)}

        tqdm.pandas(desc=f"Computing {col_name}")
        df[col_name] = df.progress_apply(_get_dos, axis=1)

    return df

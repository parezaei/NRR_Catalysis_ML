"""
dos_plots.py
============
Visualisation of site-projected and differential density of states (DOS/ΔDOS)
from VASP vasprun.xml files.

Functions
---------
plot_pdos_for_atoms      : Plot s/p/d PDOS for a list of atom indices (overlay
                           or summed).
compute_delta_pdos       : Subtract clean-surface PDOS from adsorbed PDOS to
                           obtain ΔPDOS.
gaussian_smooth          : Gaussian smoothing of a 1-D DOS curve.
plot_delta_pdos_smooth   : Plot smoothed Δ(sp)_metal and Δ(d)_metal curves.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.io.vasp import Vasprun


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_cdos(vasprun_path: str):
    """Parse a vasprun.xml and return the CompleteDos object."""
    vr = Vasprun(vasprun_path, parse_dos=True, parse_eigen=False,
                 parse_projected_eigen=False)
    return vr.complete_dos


def _dos_to_array(dos_obj) -> np.ndarray:
    """Sum spin-up and spin-down densities (or return the single spin)."""
    dens = dos_obj.densities
    if Spin.up in dens and Spin.down in dens:
        return dens[Spin.up] + dens[Spin.down]
    return list(dens.values())[0]


_ORBITAL_MAP = {
    "s":  OrbitalType.s,
    "p":  OrbitalType.p,
    "d":  OrbitalType.d,
    "sp": None,  # handled explicitly
}


# ---------------------------------------------------------------------------
# PDOS for selected atoms
# ---------------------------------------------------------------------------

def plot_pdos_for_atoms(
    vasprun_path: str,
    atom_indices: int | list[int],
    orbitals: tuple[str, ...] = ("sp", "d"),
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    combine_atoms: bool = True,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Plot s, p, d (or sp) projected DOS for a set of atom indices.

    Parameters
    ----------
    vasprun_path:
        Path to ``vasprun.xml`` (or ``.gz``).
    atom_indices:
        0-based pymatgen site indices of the atoms to project onto.
    orbitals:
        Subset of ``('s', 'p', 'd', 'sp')`` — ``'sp'`` sums s and p.
    xmin, xmax:
        Energy axis limits (eV, relative to E_F). ``None`` → auto.
    ymin, ymax:
        DOS axis limits. ``None`` → auto.
    combine_atoms:
        If ``True``, sum PDOS over all selected atoms (one set of curves).
        If ``False``, overlay each atom separately.
    title:
        Optional figure title suffix.
    save_path:
        If given, save the figure to this path (PDF or PNG).
    """
    if isinstance(atom_indices, int):
        atom_indices = [atom_indices]

    cdos = _load_cdos(vasprun_path)
    energies = cdos.energies - cdos.efermi
    structure = cdos.structure

    def _spd_for_index(idx: int) -> dict[str, np.ndarray]:
        spd = cdos.get_site_spd_dos(structure[idx])
        curves: dict[str, np.ndarray] = {}
        for orb in orbitals:
            if orb == "sp":
                arr = np.zeros_like(energies)
                for o in (OrbitalType.s, OrbitalType.p):
                    if o in spd:
                        arr += _dos_to_array(spd[o])
                curves["sp"] = arr
            elif orb in _ORBITAL_MAP and _ORBITAL_MAP[orb] in spd:
                curves[orb] = _dos_to_array(spd[_ORBITAL_MAP[orb]])
            else:
                curves[orb] = np.zeros_like(energies)
        return curves

    plt.figure(figsize=(11, 7))
    any_curve = False

    if combine_atoms:
        summed: dict[str, np.ndarray] = {o: np.zeros_like(energies) for o in orbitals}
        for idx in atom_indices:
            for orb, arr in _spd_for_index(idx).items():
                summed[orb] += arr
        labels = [f"{i}({structure[i].specie.symbol})" for i in atom_indices]
        for orb, arr in summed.items():
            if np.any(arr):
                plt.plot(energies, arr, label=f"sum {orb}", linewidth=2)
                any_curve = True
        plt.title(f"PDOS (summed): {', '.join(labels)}" + (f" — {title}" if title else ""))
    else:
        for idx in atom_indices:
            el = structure[idx].specie.symbol
            for orb, arr in _spd_for_index(idx).items():
                if np.any(arr):
                    plt.plot(energies, arr, label=f"{idx}({el}) {orb}", linewidth=1.5)
                    any_curve = True
        plt.title(f"PDOS (per atom)" + (f" — {title}" if title else ""))

    if not any_curve:
        plt.close()
        print("No PDOS data found for the selected atoms/orbitals.")
        return

    plt.axvline(0, color="k", linestyle="--", linewidth=1.0, label="$E_F$")
    plt.xlabel(r"$E - E_F$ (eV)", fontsize=14)
    plt.ylabel("DOS (states/eV)", fontsize=14)
    if xmin is not None or xmax is not None:
        plt.xlim(xmin, xmax)
    if ymin is not None or ymax is not None:
        plt.ylim(ymin, ymax)
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()


# ---------------------------------------------------------------------------
# ΔPDOS: adsorbed minus clean surface
# ---------------------------------------------------------------------------

def compute_delta_pdos(
    vasprun_ads_path: str,
    vasprun_clean_path: str,
    metal_indices: list[int],
    onto: str = "ads",
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ΔPDOS = (adsorbed) − (clean surface) for metal sp and d channels.

    Parameters
    ----------
    vasprun_ads_path:
        vasprun.xml for the adsorbed system.
    vasprun_clean_path:
        vasprun.xml for the clean surface.
    metal_indices:
        0-based site indices of the metal atoms in the active site.
        Adsorbate atoms must be excluded.
    onto:
        ``'ads'`` — interpolate clean onto adsorbed grid (default).
        ``'clean'`` — interpolate adsorbed onto clean grid.
    fill_value:
        DOS value outside the overlapping energy window.

    Returns
    -------
    tuple(energies, delta_sp, delta_d)
        Arrays restricted to the overlapping energy window.
    """
    def _load_sp_d(path: str):
        cdos = _load_cdos(path)
        E = cdos.energies - cdos.efermi
        structure = cdos.structure
        sp_total = np.zeros_like(E)
        d_total  = np.zeros_like(E)

        for idx in metal_indices:
            if idx >= len(structure):
                continue
            spd = cdos.get_site_spd_dos(structure[idx])
            for o in (OrbitalType.s, OrbitalType.p):
                if o in spd:
                    sp_total += _dos_to_array(spd[o])
            if OrbitalType.d in spd:
                d_total += _dos_to_array(spd[OrbitalType.d])

        return E, sp_total, d_total

    E_ads, sp_ads, d_ads = _load_sp_d(vasprun_ads_path)
    E_cln, sp_cln, d_cln = _load_sp_d(vasprun_clean_path)

    if onto == "ads":
        E = E_ads
        sp_cln_i = np.interp(E, E_cln, sp_cln, left=fill_value, right=fill_value)
        d_cln_i  = np.interp(E, E_cln, d_cln,  left=fill_value, right=fill_value)
        delta_sp = sp_ads - sp_cln_i
        delta_d  = d_ads  - d_cln_i
    else:
        E = E_cln
        sp_ads_i = np.interp(E, E_ads, sp_ads, left=fill_value, right=fill_value)
        d_ads_i  = np.interp(E, E_ads, d_ads,  left=fill_value, right=fill_value)
        delta_sp = sp_ads_i - sp_cln
        delta_d  = d_ads_i  - d_cln

    # Restrict to overlap window
    Emin = max(E_ads.min(), E_cln.min())
    Emax = min(E_ads.max(), E_cln.max())
    mask = (E >= Emin) & (E <= Emax)

    return E[mask], delta_sp[mask], delta_d[mask]


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------

def gaussian_smooth(
    y: np.ndarray,
    x: np.ndarray,
    sigma_eV: float = 0.10,
) -> np.ndarray:
    """Smooth a DOS curve with a Gaussian kernel of width *sigma_eV*.

    Parameters
    ----------
    y:
        DOS values.
    x:
        Energy grid (must be strictly increasing and uniform).
    sigma_eV:
        Gaussian standard deviation in eV.

    Returns
    -------
    numpy.ndarray
        Smoothed array, same length as *y*.

    Notes
    -----
    Intended for visualisation only — do not integrate the smoothed curve.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    dx = float(np.mean(np.diff(x)))
    if dx <= 0:
        raise ValueError("x must be strictly increasing.")

    sigma_pts = sigma_eV / dx
    if sigma_pts < 1e-6:
        return y.copy()

    half = int(np.ceil(4 * sigma_pts))
    grid = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (grid / sigma_pts) ** 2)
    kernel /= kernel.sum()

    return np.convolve(y, kernel, mode="same")


# ---------------------------------------------------------------------------
# Smoothed ΔPDOS plot
# ---------------------------------------------------------------------------

def plot_delta_pdos_smooth(
    vasprun_ads_path: str,
    vasprun_clean_path: str,
    metal_indices: list[int],
    save_path: str | None = None,
    onto: str = "ads",
    smooth_sigma_eV: float = 0.10,
    show_raw: bool = False,
    xlim: tuple[float, float] = (-10.0, 5.0),
    dpi: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot smoothed Δ(sp)_metal and Δ(d)_metal induced by adsorption.

    Parameters
    ----------
    vasprun_ads_path:
        vasprun.xml for the adsorbed system.
    vasprun_clean_path:
        vasprun.xml for the clean surface.
    metal_indices:
        0-based pymatgen site indices of the active-site metal atoms.
    save_path:
        If given, save the figure here (e.g. ``'delta_dos.pdf'``).
    onto:
        Target energy grid: ``'ads'`` or ``'clean'``.
    smooth_sigma_eV:
        Gaussian broadening in eV. Set ``None`` or ``0`` to disable.
    show_raw:
        If ``True``, overlay faint unsmoothed curves behind the smoothed ones.
    xlim:
        Energy axis limits.
    dpi:
        Figure resolution for saving.

    Returns
    -------
    tuple(energies, delta_sp_raw, delta_d_raw)
        Raw (unsmoothed) arrays for downstream quantitative use.
    """
    energies, delta_sp, delta_d = compute_delta_pdos(
        vasprun_ads_path, vasprun_clean_path, metal_indices, onto=onto
    )

    if smooth_sigma_eV:
        sp_plot = gaussian_smooth(delta_sp, energies, sigma_eV=smooth_sigma_eV)
        d_plot  = gaussian_smooth(delta_d,  energies, sigma_eV=smooth_sigma_eV)
        suffix  = f" (σ = {smooth_sigma_eV:.2f} eV)"
    else:
        sp_plot, d_plot, suffix = delta_sp, delta_d, ""

    plt.figure(figsize=(7, 5))

    if show_raw:
        plt.plot(energies, delta_sp, alpha=0.20, linewidth=1)
        plt.plot(energies, delta_d,  alpha=0.20, linewidth=1)

    plt.plot(energies, sp_plot, label=f"Δ(sp)_metal{suffix}", linewidth=2)
    plt.plot(energies, d_plot,  label=f"Δ(d)_metal{suffix}",  linewidth=2)

    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8, label="$E_F$")
    plt.xlim(*xlim)
    plt.xlabel(r"$E - E_F$ (eV)", fontsize=13)
    plt.ylabel("ΔDOS (states/eV)", fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()

    return energies, delta_sp, delta_d

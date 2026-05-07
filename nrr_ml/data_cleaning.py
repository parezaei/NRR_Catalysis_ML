"""
data_cleaning.py
================
Post-processing pipeline for the raw neighbour-feature DataFrames.

Steps (applied in order)
-------------------------
1. ``flatten_dos_columns``   — Unpack DOS dict columns into numeric columns.
2. ``parse_object_columns``  — Convert remaining string columns to Python types.
3. ``classify_dissociation`` — Flag rows where the molecule dissociated.
4. ``classify_adsorption``   — Flag molecular vs. atomic adsorption.
5. ``classify_orientation``  — Compute tilt angle and orientation label for
                               N₂ and N₂H.
6. ``select_output_columns`` — Reorder and subset columns for downstream use.
7. ``clean_pipeline``        — Run all steps in sequence.

All functions return copies of the input DataFrame (no in-place mutation).
"""

from __future__ import annotations

import ast
import re
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — Flatten DOS dict columns
# ---------------------------------------------------------------------------

def flatten_dos_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack ``n0_n{k}_dos`` dictionary columns into separate numeric columns.

    Each ``n0_n{k}_dos`` cell holds a dict such as::

        {'total_dband_center': -1.2, 'total_dband_sum': 3.4, ...}

    This function explodes those dicts into columns named
    ``n0_n{k}_dos_total_dband_center``, etc., and drops the originals.
    Rows with parse errors are recorded as ``n0_n{k}_dos_error`` columns
    (which are subsequently dropped by :func:`drop_error_columns`).

    Parameters
    ----------
    df:
        Raw DataFrame with DOS dict columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with DOS dicts replaced by flat numeric columns.
    """
    df = df.copy()
    dos_cols = [c for c in df.columns if re.match(r"n0_n\d+_dos$", c)]
    flat_records = []

    for _, row in df.iterrows():
        flat_row: dict = {}
        for col in dos_cols:
            raw = row.get(col, {})
            try:
                parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
            except Exception:
                parsed = {}

            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    flat_row[f"{col}_{k}"] = v[0] if isinstance(v, list) and v else v

        flat_records.append(flat_row)

    flat_df = pd.DataFrame(flat_records).fillna(0.0)
    df = pd.concat(
        [df.drop(columns=dos_cols).reset_index(drop=True), flat_df],
        axis=1,
    )
    return df


def drop_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns of the form ``n0_n{k}_dos_error`` created during parsing."""
    error_cols = [c for c in df.columns if re.match(r"n0_n\d+_dos_error", c)]
    return df.drop(columns=error_cols)


# ---------------------------------------------------------------------------
# Step 2 — Parse object columns
# ---------------------------------------------------------------------------

def parse_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string-encoded Python literals (lists, arrays, tuples) to objects.

    Handles three special cases that :func:`ast.literal_eval` cannot:

    * ``array([...])`` — converted via ``numpy``.
    * Space-separated numeric arrays (no commas) ``[1.2e-3 4.5e+6]``.
    * Anything else is left as-is.

    Parameters
    ----------
    df:
        DataFrame whose object columns may contain string-encoded values.

    Returns
    -------
    pandas.DataFrame
    """
    def _parse(value):
        if not isinstance(value, str):
            return value
        s = value.strip()
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
        if "array" in s:
            try:
                safe = re.sub(r"\barray\s*\(", "np.array(", s.replace("\n", " "))
                return eval(safe, {"np": np})  # noqa: S307
            except Exception:
                return value
        if s.startswith("[") and s.endswith("]") and "," not in s:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if nums:
                return np.array([float(n) for n in nums])
        return value

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        if df[col].apply(lambda x: isinstance(x, str)).any():
            try:
                df[col] = df[col].apply(_parse)
            except Exception:
                pass
    return df


# ---------------------------------------------------------------------------
# Step 3 — Dissociation flags
# ---------------------------------------------------------------------------

def classify_dissociation(
    df_dict: dict[str, pd.DataFrame],
    thresholds: dict[str, dict] | None = None,
) -> dict[str, pd.DataFrame]:
    """Add an ``is_dissociated`` column to each molecule DataFrame.

    Default dissociation thresholds:

    ========  ======================================
    Molecule  Criterion
    ========  ======================================
    H₂        post_H-H  > 1.00 Å
    N₂        post_N-N  > 1.30 Å
    N₂H       post_N-N  > 1.45 Å or post_N-H1 > 1.25 Å
    NH₃       any post_N-Hk > 1.25 Å (k = 1,2,3)
    ========  ======================================

    Parameters
    ----------
    df_dict:
        ``{'N2': df, 'N2H': df, 'NH3': df}``.
    thresholds:
        Optional override dict; same structure as defaults.

    Returns
    -------
    dict[str, pandas.DataFrame]
    """
    defaults: dict[str, dict] = {
        "N2":  {"N-N": 1.30},
        "N2H": {"N-N": 1.45, "N-H1": 1.25},
        "NH3": {"N-H1": 1.25, "N-H2": 1.25, "N-H3": 1.25},
    }
    if thresholds:
        defaults.update(thresholds)

    out = {}
    for mol, df in df_dict.items():
        df = df.copy()
        thresh = defaults.get(mol, {})

        if mol == "N2":
            df["is_dissociated"] = (df["post_N-N"] > thresh["N-N"]).astype(int)
        elif mol == "N2H":
            df["is_dissociated"] = (
                (df["post_N-N"] > thresh["N-N"]) | (df["post_N-H1"] > thresh["N-H1"])
            ).astype(int)
        elif mol == "NH3":
            df["is_dissociated"] = (
                (df["post_N-H1"] > thresh["N-H1"])
                | (df["post_N-H2"] > thresh["N-H2"])
                | (df["post_N-H3"] > thresh["N-H3"])
            ).astype(int)
        else:
            df["is_dissociated"] = 0

        out[mol] = df

    return out


# ---------------------------------------------------------------------------
# Step 4 — Adsorption classification
# ---------------------------------------------------------------------------

def classify_adsorption(
    df_dict: dict[str, pd.DataFrame],
    distance_threshold: float = 2.5,
) -> dict[str, pd.DataFrame]:
    """Add ``Mol/Atom_is_adsorbed``, ``Mol_is_adsorbed``, and
    ``atomic_is_adsorbed`` columns.

    An adsorption event is detected when any adsorbate atom is within
    *distance_threshold* Å of the nearest metal atom (``metal0``).

    Parameters
    ----------
    df_dict:
        Molecule DataFrames, each already containing ``is_dissociated``.
    distance_threshold:
        Maximum metal–adsorbate distance (Å) to count as adsorbed.

    Returns
    -------
    dict[str, pandas.DataFrame]
    """
    _dist_cols: dict[str, list[str]] = {
        "N2":  ["N0-metal0_distance", "N1-metal0_distance"],
        "N2H": ["N0-metal0_distance", "N1-metal0_distance", "H0-metal0_distance"],
        "NH3": ["N0-metal0_distance", "H0-metal0_distance",
                "H1-metal0_distance", "H2-metal0_distance"],
    }

    out = {}
    for mol, df in df_dict.items():
        df = df.copy()
        cols = _dist_cols.get(mol, [])
        existing = [c for c in cols if c in df.columns]

        is_close = df[existing].apply(pd.to_numeric, errors="coerce").lt(distance_threshold).any(axis=1)
        df["Mol/Atom_is_adsorbed"] = is_close.astype(int)
        df["Mol_is_adsorbed"] = ((df["is_dissociated"] == 0) & is_close).astype(int)
        df["atomic_is_adsorbed"] = ((df["is_dissociated"] == 1) & is_close).astype(int)

        out[mol] = df

    return out


# ---------------------------------------------------------------------------
# Step 5 — Orientation classification (N₂ / N₂H)
# ---------------------------------------------------------------------------

def _classify_single_orientation(
    mol_coords: list[tuple[str, np.ndarray]],
    atom_pair: tuple[str, str] = ("N", "N"),
    vertical_thresh: float = 20.0,
    horizontal_thresh: float = 20.0,
) -> dict[str, float | str]:
    """Compute bond vector angle relative to the surface normal (z-axis).

    Parameters
    ----------
    mol_coords:
        List of ``(element, np.ndarray)`` tuples.
    atom_pair:
        Two element symbols that define the bond to measure.
    vertical_thresh, horizontal_thresh:
        Degree tolerances for 'vertical' (≤ thresh from z) and 'horizontal'
        (≤ thresh from xy-plane).

    Returns
    -------
    dict with keys ``'angle'`` (float, degrees) and ``'orientation'`` (str).
    """
    found: list[np.ndarray] = []
    for atype, coord in mol_coords:
        if atype == atom_pair[len(found)] and len(found) < 2:
            found.append(np.asarray(coord))
        if len(found) == 2:
            break

    if len(found) < 2:
        return {"angle": np.nan, "orientation": "unknown"}

    vec = found[1] - found[0]
    vec = vec / np.linalg.norm(vec)
    z_axis = np.array([0.0, 0.0, 1.0])
    angle = float(np.degrees(np.arccos(np.clip(np.dot(vec, z_axis), -1.0, 1.0))))

    if angle <= vertical_thresh or abs(angle - 180) <= vertical_thresh:
        orientation = "vertical"
    elif abs(angle - 90) <= horizontal_thresh:
        orientation = "horizontal"
    else:
        orientation = "tilted"

    return {"angle": angle, "orientation": orientation}


def classify_orientation(
    df_dict: dict[str, pd.DataFrame],
    mols: tuple[str, ...] = ("N2", "N2H"),
) -> dict[str, pd.DataFrame]:
    """Add ``angle`` and ``orientation`` columns to N₂ and N₂H DataFrames.

    Only rows where ``Mol_is_adsorbed == 1`` are processed; the rest receive
    ``NaN`` / ``None``.

    Parameters
    ----------
    df_dict:
        Molecule DataFrames.
    mols:
        Which molecules to process (default N₂ and N₂H).

    Returns
    -------
    dict[str, pandas.DataFrame]
    """
    out = dict(df_dict)
    for mol in mols:
        if mol not in df_dict:
            continue
        df = df_dict[mol].copy()
        mask = df["Mol_is_adsorbed"] == 1
        orient_data = (
            df.loc[mask, "mol_output_all_coords"]
            .apply(lambda c: pd.Series(_classify_single_orientation(c)))
        )
        df.loc[mask, "angle"] = orient_data["angle"].values
        df.loc[mask, "orientation"] = orient_data["orientation"].values
        out[mol] = df
    return out


# ---------------------------------------------------------------------------
# Step 6 — Column selection
# ---------------------------------------------------------------------------

def select_output_columns(
    df: pd.DataFrame,
    mol: str,
    n_neighbors: int = 20,
) -> pd.DataFrame:
    """Reorder and subset columns to produce the final clean DataFrame.

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    mol:
        Molecule name (``'N2'``, ``'N2H'``, or ``'NH3'``).
    n_neighbors:
        Maximum number of neighbour columns to retain.

    Returns
    -------
    pandas.DataFrame
    """
    initial = [
        "material", "ad_site_number", "crystal_struct", "sym_group",
        "mol", "mol_elecneg",
        "pre_N-N", "pre_N-H1", "pre_N-H2", "pre_N-H3",
        "mol_cent_coord", "mol_input_all_coords",
    ]
    neighbor_cols = [
        c for i in range(n_neighbors + 1)
        for c in df.columns if c.startswith(f"n{i}_")
    ]
    dos_cols = [c for c in df.columns if re.match(r"n0_n\d+_dos", c)]

    ending = [
        "mol_output_all_coords", "is_dissociated", "Mol_is_adsorbed",
        "atomic_is_adsorbed",
    ]
    if mol in ("N2", "N2H"):
        ending += ["angle", "orientation"]
    ending.append("ads_e")

    ordered = [c for c in initial + neighbor_cols + dos_cols + ending if c in df.columns]
    return df[ordered]


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def clean_pipeline(
    df_dict: dict[str, pd.DataFrame],
    adsorption_threshold: float = 2.5,
) -> dict[str, pd.DataFrame]:
    """Apply all cleaning steps in sequence.

    Steps applied:

    1. Flatten DOS dict columns.
    2. Drop error columns.
    3. Parse remaining object columns.
    4. Classify dissociation.
    5. Classify adsorption.
    6. Classify orientation (N₂, N₂H only).
    7. Drop index columns.
    8. Select and reorder output columns.

    Parameters
    ----------
    df_dict:
        ``{'N2': df, 'N2H': df, 'NH3': df}`` — raw expanded DataFrames.
    adsorption_threshold:
        Distance threshold (Å) for adsorption classification.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Cleaned DataFrames keyed by molecule name.
    """
    # Steps 1–3: parse
    parsed = {}
    for mol, df in df_dict.items():
        df = flatten_dos_columns(df)
        df = drop_error_columns(df)
        df = parse_object_columns(df)
        parsed[mol] = df

    # Steps 4–6: classify
    result = classify_dissociation(parsed)
    result = classify_adsorption(result, distance_threshold=adsorption_threshold)
    result = classify_orientation(result)

    # Step 7: drop index columns
    for mol in list(result):
        df = result[mol]
        index_cols = [c for c in df.columns if "index" in c.lower()]
        result[mol] = df.drop(columns=index_cols, errors="ignore")

    # Step 8: select columns
    for mol in list(result):
        result[mol] = select_output_columns(result[mol], mol)

    return result

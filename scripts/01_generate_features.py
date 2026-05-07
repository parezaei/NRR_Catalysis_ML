"""
01_generate_features.py
=======================
Compute DOS band-centre / filling-fraction features for every adsorption site
and append them as ``n0_n{k}_dos`` columns to the raw expanded DataFrames.

Usage
-----
    python scripts/01_generate_features.py

Edit the paths at the top of this script before running.
"""

import pandas as pd
from nrr_ml.features import add_dos_features

# ── Paths — edit these ────────────────────────────────────────────────────────
RAW_DFS = {
    "N2":  "./dataframes/raw_df_N2_bandcenter_expanded_v5.csv",
    "N2H": "./dataframes/raw_df_N2H_bandcenter_expanded_v5.csv",
    "NH3": "./dataframes/raw_df_NH3_bandcenter_expanded_v5.csv",
}

OUT_DFS = {
    "N2":  "./dataframes/raw_df_N2_bandcenter_expanded_v5_dos.csv",
    "N2H": "./dataframes/raw_df_N2H_bandcenter_expanded_v5_dos.csv",
    "NH3": "./dataframes/raw_df_NH3_bandcenter_expanded_v5_dos.csv",
}

# Root directory containing per-person, per-material sub-folders with
# the surface vasprun.xml files:
BASE_VASPRUN_DIR = "/path/to/scf_bader_DOS_surf"

MAX_NEIGHBORS = 20
EMIN, EMAX    = -10.0, 10.0

# ── Run ───────────────────────────────────────────────────────────────────────
for mol, csv_path in RAW_DFS.items():
    print(f"\n{'='*60}\nProcessing {mol}\n{'='*60}")
    df = pd.read_csv(csv_path)
    df = add_dos_features(
        df,
        base_vasprun_dir=BASE_VASPRUN_DIR,
        max_neighbors=MAX_NEIGHBORS,
        emin=EMIN,
        emax=EMAX,
    )
    df.to_csv(OUT_DFS[mol], index=False)
    print(f"Saved → {OUT_DFS[mol]}  ({df.shape})")

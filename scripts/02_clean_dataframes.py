"""
02_clean_dataframes.py
======================
Apply the full data-cleaning pipeline to the raw expanded DataFrames and save
the cleaned versions used as input to the ML pipeline.

Steps applied
-------------
1. Flatten DOS dict columns into numeric columns.
2. Parse object-typed columns (lists, arrays, tuples).
3. Classify dissociated / adsorbed / molecular-adsorbed / atomic-adsorbed.
4. Classify N₂ and N₂H orientation (vertical / horizontal / tilted).
5. Select and reorder output columns.

Usage
-----
    python scripts/02_clean_dataframes.py
"""

import pandas as pd
from nrr_ml.data_cleaning import clean_pipeline

# ── Paths — edit these ────────────────────────────────────────────────────────
RAW_DFS = {
    "N2":  "./dataframes/raw_df_N2_bandcenter_expanded_v5.csv",
    "N2H": "./dataframes/raw_df_N2H_bandcenter_expanded_v5.csv",
    "NH3": "./dataframes/raw_df_NH3_bandcenter_expanded_v5.csv",
}

OUT_DFS = {
    "N2":  "./dataframes/input_df_N2_bandcenter_v1.csv",
    "N2H": "./dataframes/input_df_N2H_bandcenter_v1.csv",
    "NH3": "./dataframes/input_df_NH3_bandcenter_v1.csv",
}

ADSORPTION_THRESHOLD = 2.5  # Å

# ── Run ───────────────────────────────────────────────────────────────────────
print("Loading raw DataFrames …")
df_dict = {mol: pd.read_csv(path) for mol, path in RAW_DFS.items()}

print("Running cleaning pipeline …")
cleaned = clean_pipeline(df_dict, adsorption_threshold=ADSORPTION_THRESHOLD)

for mol, df in cleaned.items():
    path = OUT_DFS[mol]
    df.to_csv(path, index=False)
    adsorbed = df["Mol_is_adsorbed"].sum() if "Mol_is_adsorbed" in df.columns else "?"
    print(f"  {mol}: {df.shape[0]} rows, {adsorbed} molecularly adsorbed → {path}")

print("\nDone.")

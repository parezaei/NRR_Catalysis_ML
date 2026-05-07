"""
03_run_ml_pipeline.py
=====================
Run the neighbour-sweep ML pipeline for N₂, N₂H, and NH₃ adsorption energy
prediction, generate all publication figures, and save fitted models.

Workflow
--------
1. Load cleaned input DataFrames.
2. For each molecule:
   a. Run :func:`nrr_ml.ml_pipeline.run_pipeline` — sweeps neighbours 0..N,
      cross-validates 10+ regressors, selects the best (model, n) pair.
   b. Save the fitted model and results dict with ``pickle``.
   c. Generate CV sweep, parity, SHAP bar, beeswarm, and dependence plots.

Usage
-----
    python scripts/03_run_ml_pipeline.py

Edit the configuration block at the top before running.
"""

import os
import pickle
import pandas as pd

from nrr_ml.ml_pipeline import (
    run_pipeline,
    plot_cv_sweep,
    plot_train_vs_cv,
    plot_heatmap,
    plot_parity,
    plot_shap,
)

# ── Configuration — edit these ────────────────────────────────────────────────
INPUT_DFS = {
    "N2":  "./dataframes/input_df_N2_bandcenter_v1.csv",
    "N2H": "./dataframes/input_df_N2H_bandcenter_v1.csv",
    "NH3": "./dataframes/input_df_NH3_bandcenter_v1.csv",
}

SAVE_DIR       = "./paper_files/"          # figures and pickles saved here
OUTLIER_METHOD = "z3"                      # 'z3' | 'z2' | 'iqr' | 'physical' | None
N_FEATURES     = 20                        # SelectKBest k
N_NEIGHBORS    = 21                        # neighbour shells to sweep (0 … N-1)
RANDOM_STATE   = 42

# ── Run ───────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)

for mol, csv_path in INPUT_DFS.items():
    print(f"\n{'='*70}\nMolecule: {mol}\n{'='*70}")

    df_raw = pd.read_csv(csv_path)

    results = run_pipeline(
        df_raw            = df_raw,
        outlier_method    = OUTLIER_METHOD,
        n_select_features = N_FEATURES,
        tested_n_neighbors = N_NEIGHBORS,
        random_state      = RANDOM_STATE,
    )

    # Save results (model + arrays)
    pkl_path = os.path.join(
        SAVE_DIR,
        f"results_{mol}_{OUTLIER_METHOD}_f{N_FEATURES}_n{N_NEIGHBORS}.pkl",
    )
    with open(pkl_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"Results saved → {pkl_path}")

    # Generate figures
    base = os.path.join(SAVE_DIR, f"{mol}_f{N_FEATURES}_outlier{OUTLIER_METHOD}_n{N_NEIGHBORS}")

    plot_cv_sweep(results,    filename=f"{base}_cv_sweep.pdf")
    plot_train_vs_cv(results, filename=f"{base}_train_vs_cv.pdf")
    plot_heatmap(results,     filename=f"{base}_heatmap.pdf")
    plot_parity(results,      filename=f"{base}_parity.pdf")
    plot_shap(
        results,
        filename_bar        = f"{base}_shap_bar.pdf",
        filename_beeswarm   = f"{base}_shap_beeswarm.pdf",
        filename_dependence = f"{base}_shap_dependence.pdf",
        n_top               = N_FEATURES,
    )

    print(
        f"\n  Best model : {results['best_model_nm']}"
        f"\n  Best n     : {results['best_k']}"
        f"\n  CV  R²     : {results['best_cv_r2']:.4f}"
        f"\n  Test R²    : {results['test_r2']:.4f}"
        f"\n  Test MAE   : {results['test_mae']:.4f} eV"
    )

print("\nAll done.")

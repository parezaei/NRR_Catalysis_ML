# ML-Accelerated Prediction of NRR Intermediate Adsorption Energies on Alloy Surfaces

[![Lab](https://img.shields.io/badge/Lab-Insilico%20Matters%20Laboratory-blue)](https://github.com/insilicomatters)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)

**Author:** Parastoo Agharezaei  
**Supervisor:** Prof. Kulbir K. Ghuman  
**Laboratory:** [Insilico Matters Laboratory (IML)](https://github.com/insilicomatters) · INRS-EMT

---

## Overview

This repository contains the complete code used to produce the results in our paper on machine-learning prediction of adsorption energies for nitrogen reduction reaction (NRR) intermediates — **N₂, N₂H, and NH₃** — on transition metal alloy surfaces.

We combine density functional theory (DFT) calculations (VASP) with a systematic machine learning pipeline to predict adsorption energies from local structural and electronic descriptors extracted directly from the DFT outputs.

### Feature set

| Category | Features |
|---|---|
| Atomic properties | Electronegativity, electron affinity, ionisation energy, atomic radius, covalent radius, atomic density, Bader charge |
| Structural | Distance to nearest metal atoms (up to 21 neighbours), crystal structure type, space group |
| Electronic | s/p/d band centres (ε) and filling fractions (f) from DFT PDOS, both local (selected-atom) and total |
| Shell statistics | Mean, std, min, max, range, and distance-weighted mean of all properties over the neighbour shell |

---

## Repository structure

```
NRR_Catalysis_ML/
├── nrr_ml/                      # Python package
│   ├── vasp_io.py               # VASP file parsing (OUTCAR, CONTCAR, ACF.dat, POTCAR)
│   ├── structure.py             # Supercell expansion, neighbour finding, Bader charge mapping
│   ├── features.py              # DOS band-centre / filling-fraction extraction
│   ├── data_cleaning.py         # Adsorption classification, orientation analysis, column selection
│   ├── ml_pipeline.py           # Full ML pipeline: neighbour sweep, cross-validation, SHAP
│   ├── dos_plots.py             # PDOS and ΔPDOS visualisation
│   └── ml_plots.py              # Parity plots, model comparison, SHAP beeswarm
├── scripts/
│   ├── 01_generate_features.py  # Add DOS features to raw DataFrames
│   ├── 02_clean_dataframes.py   # Run the full data-cleaning pipeline
│   └── 03_run_ml_pipeline.py    # Train models, evaluate, generate figures
├── requirements.txt
└── README.md
```

---

## Workflow

```
VASP DFT calculations
        │
        ▼
scripts/01_generate_features.py
   └─ Parse vasprun.xml with pymatgen → compute s/p/d band centres & filling
      fractions for each cumulative neighbour shell → save to CSV
        │
        ▼
scripts/02_clean_dataframes.py
   └─ Flatten DOS columns · classify dissociation / adsorption mode
      · compute N₂/N₂H tilt angle · select output columns → save to CSV
        │
        ▼
scripts/03_run_ml_pipeline.py
   └─ Neighbour sweep (0 … 20) × 10+ regressors × repeated 5-fold CV
      → select best (model, n) → test-set evaluation → SHAP plots → save pkl
```

---

## ML pipeline highlights (`nrr_ml/ml_pipeline.py`)

| Step | Detail |
|---|---|
| Outlier removal | z-score (±3σ), IQR, physical bounds, or none |
| Feature engineering | Nearest-atom properties + distance-weighted shell statistics + DOS features |
| Feature selection | Variance threshold → drop high-correlation pairs (|r| > 0.95) → SelectKBest (F-regression) |
| Cross-validation | 5-fold × 5 repeats (RepeatedKFold) |
| Models compared | Ridge, Lasso, ElasticNet, Huber, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, SVR, Kernel Ridge, KNN, MLP, **XGBoost** |
| Model selection | Best CV R² across all (model, neighbour-level) combinations |
| Explainability | SHAP TreeExplainer / LinearExplainer / KernelExplainer — bar, beeswarm, dependence plots |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/arzpa/NRR_Catalysis_ML.git
cd NRR_Catalysis_ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit paths in scripts (BASE_VASPRUN_DIR, INPUT_DFS, SAVE_DIR)

# 4. Generate DOS features
python scripts/01_generate_features.py

# 5. Clean DataFrames
python scripts/02_clean_dataframes.py

# 6. Run ML pipeline
python scripts/03_run_ml_pipeline.py
```

Or import individual modules directly:

```python
from nrr_ml.ml_pipeline import run_pipeline, plot_parity
import pandas as pd

df = pd.read_csv("dataframes/input_df_N2_bandcenter_v1.csv")
results = run_pipeline(df, outlier_method="z3", n_select_features=20)
plot_parity(results, filename="N2_parity.pdf")
```

---

## Data and figures availability

The cleaned input datasets and all paper figures will be added to this repository upon publication.

Raw DFT output files (OUTCAR, vasprun.xml, ACF.dat, etc.) are available on request — please open an issue or contact the authors.

---

## Citation

If you use this code, please cite our paper (citation to be added upon publication).

---

## License

MIT © Parastoo Agharezaei, Insilico Matters Laboratory, INRS-EMT

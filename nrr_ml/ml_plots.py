"""
ml_plots.py
===========
Publication-quality visualisation of ML results from the neighbour-sweep
pipeline (``ml_pipeline.run_pipeline`` / ``ml_pipeline.run_neighbor_sweep_models``).

Functions
---------
plot_pred_vs_actual      : Train/test parity plots with R² and MAE annotations.
plot_top_models_metrics  : Bar chart of MAE, RMSE, R² for the top-k models.
plot_best_per_n          : R² / MAE vs. number of neighbours (best model per n).
explain_with_shap        : SHAP bar and beeswarm plots with clean feature labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Shared abbreviation maps
# ---------------------------------------------------------------------------

MODEL_ABBR: dict[str, str] = {
    "LinearRegression":        "LR",
    "Ridge":                   "Ridge",
    "Lasso":                   "Lasso",
    "ElasticNet":              "EN",
    "HuberRegressor":          "Huber",
    "DecisionTreeRegressor":   "DT",
    "RandomForestRegressor":   "RF",
    "GradientBoostingRegressor": "GB",
    "ExtraTreesRegressor":     "ET",
    "SVR":                     "SVR",
    "KernelRidge":             "KR",
    "KNeighborsRegressor":     "KNN",
    "MLPRegressor":            "MLP",
    "XGBRegressor":            "XGB",
    "TabPFNRegressor":         "TabPFN",
}

_SHAP_PROP_MAP: dict[str, str] = {
    "electroneg":     "χ",
    "elect_affinity": "EA",
    "atomic_radius":  "r",
    "cov_rad":        "rc",
    "atom_density":   "ρ",
    "charge":         "q",
}


def _short_feature_name(colname: str) -> str:
    """Map a verbose feature column name to a compact LaTeX-style label."""
    import re
    for prop, sym in _SHAP_PROP_MAP.items():
        if f"_{prop}_" in colname:
            if colname.startswith("wavg_"):
                return f"$WA_{{{sym},\\mathrm{{local}}}}$"
            if colname.startswith("avg_"):
                return f"$A_{{{sym},\\mathrm{{local}}}}$"

    if "dos" in colname:
        parts = colname.split("_")
        try:
            sel  = parts[3]
            band = parts[4][0]
            stat = parts[-1]
            loc  = r"\mathrm{loc}" if sel == "selected" else r"\mathrm{tot}"
            sym  = r"\varepsilon" if stat == "center" else "f"
            return f"${sym}_{{{band},{loc}}}$"
        except IndexError:
            pass

    return colname


# ---------------------------------------------------------------------------
# Parity plot
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(
    results: dict,
    title_prefix: str = "",
    first_set: str = "trainval",
    fixed_lims: tuple[float, float] | None = None,
    dot_size: int = 120,
    label_size: int = 20,
    r2_size: int = 18,
    legend_size: int = 18,
    save_path: str | None = None,
) -> None:
    """Two-panel parity plot: training (or validation) set and test set.

    Parameters
    ----------
    results:
        Dict returned by :func:`ml_pipeline.run_neighbor_sweep_models` after
        re-running on the best model (must contain ``X_trainval``,
        ``trainval_df``, ``X_val``, ``y_val_pred``, ``y_test``,
        ``y_test_pred``, ``best_pipeline``, ``best_model_name``, ``best_n``).
    title_prefix:
        Molecule label used in titles and filenames (e.g. ``'N2'``).
    first_set:
        ``'trainval'`` or ``'val'`` — which data to show in the left panel.
    fixed_lims:
        Shared axis limits ``(lo, hi)``. Computed automatically if ``None``.
    dot_size, label_size, r2_size, legend_size:
        Matplotlib styling parameters.
    save_path:
        If given, save figure here.
    """
    target = results["config"]["target"]

    X_tv   = results["X_trainval"]
    y_tv   = pd.to_numeric(results["trainval_df"][target], errors="coerce").loc[X_tv.index]
    y_tv_p = results["best_pipeline"].predict(X_tv)

    X_val  = results["X_val"]
    y_val  = pd.to_numeric(results["trainval_df"][target], errors="coerce").loc[X_val.index]
    y_val_p = results["y_val_pred"]

    y_test  = results["y_test"]
    y_test_p = results["y_test_pred"]
    model_name = results["best_model_name"]
    best_n     = results["best_n"]

    if first_set == "val":
        y_L, y_L_p, label_L, color_L = y_val, y_val_p, "Validation", "steelblue"
    else:
        y_L, y_L_p, label_L, color_L = y_tv, y_tv_p, "Train", "steelblue"

    r2_L  = r2_score(y_L, y_L_p)
    mae_L = mean_absolute_error(y_L, y_L_p)
    r2_T  = r2_score(y_test, y_test_p)
    mae_T = mean_absolute_error(y_test, y_test_p)

    if fixed_lims:
        lims = list(fixed_lims)
    else:
        all_y = pd.concat([y_L, pd.Series(y_test)])
        margin = 0.05 * (all_y.max() - all_y.min())
        lims = [all_y.min() - margin, all_y.max() + margin]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    def _panel(ax, y_true, y_pred, label, color, r2, mae):
        sns.scatterplot(x=y_true, y=y_pred, ax=ax,
                        s=dot_size, color=color, edgecolor="black",
                        linewidth=0.9, alpha=0.8, label=label)
        ax.plot(lims, lims, "r--", lw=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("DFT $E_{ads}$ (eV)", fontsize=label_size, fontweight="bold")
        ax.set_ylabel("Predicted $E_{ads}$ (eV)", fontsize=label_size, fontweight="bold")
        ax.set_title(f"{title_prefix} {label}\n{model_name}, N={best_n}", fontsize=14, pad=18)
        ax.tick_params(axis="both", labelsize=label_size - 2)
        ax.legend(loc="lower right", frameon=True, fontsize=legend_size)
        ax.text(0.05, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f} eV",
                transform=ax.transAxes, fontsize=r2_size, verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    _panel(axes[0], y_L,    y_L_p,   label_L,  color_L,  r2_L, mae_L)
    _panel(axes[1], y_test, y_test_p, "Test", "darkorange", r2_T, mae_T)

    plt.suptitle(f"Predicted vs Actual — {title_prefix}", fontsize=16, weight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Top-k model bar chart
# ---------------------------------------------------------------------------

def plot_top_models_metrics(
    results: dict,
    mol_name: str = "",
    top_k: int = 3,
    exclude_models: tuple[str, ...] = ("TabPFNRegressor", "MLPRegressor"),
    save_path: str | None = None,
) -> None:
    """Bar chart of MAE, RMSE, R² for the top *top_k* models at the best n.

    Parameters
    ----------
    results:
        Dict returned by :func:`ml_pipeline.run_neighbor_sweep_models`.
    mol_name:
        Molecule label for the title and filename.
    top_k:
        Number of models to show.
    exclude_models:
        Model names to skip (TabPFN and MLP are excluded by default because
        they are slow / unstable in cross-validation).
    save_path:
        If given, save the figure here.
    """
    best_n, _, _ = results["best_on_val"]
    val_at_n = results["val_results"][best_n]

    rows = [
        {
            "model": MODEL_ABBR.get(name, name),
            "R2":   m["r2"],
            "RMSE": m["rmse"],
            "MAE":  m["mae"],
        }
        for name, m in val_at_n.items()
        if "r2" in m and name not in exclude_models
    ]

    df_top = (
        pd.DataFrame(rows)
        .sort_values("MAE", ascending=True)
        .head(top_k)
    )
    df_melt = df_top.melt(id_vars="model", value_vars=["MAE", "RMSE", "R2"],
                          var_name="Metric", value_name="Value")

    palette = {"MAE": "#e74c3c", "RMSE": "#2ecc71", "R2": "#f9a7b0"}

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melt, x="model", y="Value", hue="Metric",
                     palette=palette, width=0.6,
                     edgecolor="black", linewidth=1.2)

    for p in ax.patches:
        h = p.get_height()
        if h > 1e-6:
            ax.annotate(f"{h:.3f}",
                        (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=13, weight="bold")

    plt.ylim(0, 1)
    plt.title(f"Top {top_k} Models at n={best_n} (Validation) — {mol_name}",
              fontsize=14, weight="bold", pad=18)
    plt.xlabel("Model", fontsize=20, fontweight="bold")
    plt.ylabel("Metric Value", fontsize=20, fontweight="bold")
    plt.xticks(rotation=0, ha="center", fontsize=17)
    plt.yticks(fontsize=17)
    plt.legend(fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# R² / MAE vs. number of neighbours
# ---------------------------------------------------------------------------

def plot_best_per_n(
    results: dict,
    n_min: int = 0,
    n_max: int = 20,
    exclude_models: tuple[str, ...] | None = None,
    mol_name: str = "",
    save_dir: str | None = None,
) -> None:
    """Plot R² and MAE across neighbour levels, labelling the best model at each n.

    Parameters
    ----------
    results:
        Dict returned by :func:`ml_pipeline.run_neighbor_sweep_models`.
    n_min, n_max:
        Range of neighbour levels to plot.
    exclude_models:
        Model names to skip.
    mol_name:
        Molecule label for titles and filenames.
    save_dir:
        Directory in which to save figures. ``None`` → do not save.
    """
    if exclude_models is None:
        exclude_models = ()

    ns, r2_vals, mae_vals, models_used = [], [], [], []

    for n in range(n_min, n_max + 1):
        mres = results.get("val_results", {}).get(n, {})
        best_name = best_r2 = best_mae = None
        for name, m in mres.items():
            if name in exclude_models or "r2" not in m:
                continue
            r2v, maev = m["r2"], m["mae"]
            if best_r2 is None or r2v > best_r2 or (
                abs(r2v - best_r2) < 1e-6 and maev < best_mae
            ):
                best_name, best_r2, best_mae = name, r2v, maev

        if best_name:
            ns.append(n + 1)
            r2_vals.append(best_r2)
            mae_vals.append(best_mae)
            models_used.append(MODEL_ABBR.get(best_name, best_name))

    def _annotate(ax, xs, ys, labels):
        yrange = max(ys) - min(ys) if len(ys) > 1 else 1
        offset = 0.12 * yrange
        for i, (x, y, m) in enumerate(zip(xs, ys, labels)):
            va = "bottom" if i % 2 == 0 else "top"
            dy = offset if i % 2 == 0 else -offset
            ax.text(x, y + dy, m, ha="center", va=va, fontsize=12)

    for metric, vals, ylabel, color, fname_suffix in [
        ("R²",  r2_vals,  "R² (validation)",  "blue",  "cvR2VSneighbor"),
        ("MAE", mae_vals, "MAE (validation, eV)", "red", "cvMAEVSneighbor"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ns, vals, marker="o", color=color)
        _annotate(ax, ns, vals, models_used)
        ax.set_xlabel("Number of neighbours", fontsize=18, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")
        ax.set_title(f"Best model {metric} vs. neighbours — {mol_name}",
                     fontsize=14, pad=22)
        ax.set_xticks(range(1, n_max + 2))
        margin = 0.1 * (max(vals) - min(vals)) if len(vals) > 1 else 0.1
        ax.set_ylim(min(vals) - margin, max(vals) + margin)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.35)
        plt.tight_layout()

        if save_dir:
            import os
            plt.savefig(
                os.path.join(save_dir, f"{mol_name}_{fname_suffix}_adsorbedonly.pdf"),
                dpi=600, bbox_inches="tight",
            )
        plt.show()


# ---------------------------------------------------------------------------
# SHAP bar + beeswarm
# ---------------------------------------------------------------------------

def explain_with_shap(
    results: dict,
    n_sample: int = 200,
    n_background: int = 100,
    mol_name: str = "molecule",
    save_dir: str | None = None,
) -> tuple:
    """Generate SHAP bar and beeswarm plots for the best fitted model.

    Uses :class:`shap.TreeExplainer` for tree-based models,
    :class:`shap.LinearExplainer` for linear models, and
    :class:`shap.KernelExplainer` for everything else.

    Parameters
    ----------
    results:
        Dict returned by :func:`ml_pipeline.run_neighbor_sweep_models` after
        the best pipeline has been fitted (must contain ``best_pipeline``,
        ``X_trainval``, ``best_model_name``, ``best_n``).
    n_sample:
        Number of training samples for SHAP computation.
    n_background:
        Background samples for KernelExplainer.
    mol_name:
        Molecule label for titles and filenames.
    save_dir:
        Directory in which to save figures.

    Returns
    -------
    tuple(explainer, shap_values)
    """
    best_pipe = results["best_pipeline"]
    model = best_pipe.named_steps["model"]
    X = results["X_trainval"]

    X_sample = X.sample(min(n_sample, len(X)), random_state=42)
    rename = {c: _short_feature_name(c) for c in X_sample.columns}
    X_sample = X_sample.rename(columns=rename)

    _TREE_MODELS  = (RandomForestRegressor, GradientBoostingRegressor,
                     ExtraTreesRegressor, DecisionTreeRegressor)
    _LINEAR_MODELS = (LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor)

    if _HAS_XGB:
        _TREE_MODELS = _TREE_MODELS + (XGBRegressor,)

    if isinstance(model, _TREE_MODELS):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
    elif isinstance(model, _LINEAR_MODELS):
        explainer = shap.LinearExplainer(model, X)
        shap_vals = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X, min(n_background, len(X)), random_state=42)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_vals = explainer.shap_values(X_sample)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    best_n   = results.get("best_n", "?")
    model_nm = model.__class__.__name__

    # --- Bar plot ---
    plt.figure()
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False, color="seagreen")
    ax = plt.gca()
    xmax = mean_abs.max() * 1.15
    ax.set_xlim(0, xmax)

    order = np.argsort(mean_abs)[::-1]
    sorted_vals = mean_abs[order][::-1][: len(ax.patches)]
    for bar, val in zip(ax.patches, sorted_vals):
        ax.text(bar.get_width() + 0.02 * xmax,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=11)
        bar.set_edgecolor("black"); bar.set_linewidth(0.7)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_edgecolor("black")
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Mean |SHAP value|", fontsize=14, fontweight="bold")
    plt.title(f"Feature Importance — {mol_name} | {model_nm} | n={best_n}",
              fontsize=14, pad=22)
    plt.tight_layout()
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f"{mol_name}_SHAP_bar.pdf"),
                    dpi=600, bbox_inches="tight")
    plt.show()

    # --- Beeswarm (non-zero features only) ---
    nonzero = mean_abs > 1e-12
    sv_filt = shap_vals[:, nonzero]
    X_filt  = X_sample.loc[:, nonzero]

    plt.figure()
    shap.summary_plot(sv_filt, X_filt, plot_type="dot", show=False,
                      color_bar=True, color="coolwarm")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_edgecolor("black")
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=14, fontweight="bold")
    plt.title(f"SHAP Beeswarm — {mol_name} | {model_nm} | n={best_n}",
              fontsize=14, pad=22)
    plt.tight_layout()
    if save_dir:
        import os
        plt.savefig(os.path.join(save_dir, f"{mol_name}_SHAP_beeswarm.pdf"),
                    dpi=600, bbox_inches="tight")
    plt.show()

    return explainer, shap_vals

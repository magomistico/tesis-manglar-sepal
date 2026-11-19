"""Gráficos de calidad publicación (300 DPI) para los capítulos de Resultados de la tesis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_thesis_style() -> None:
    """Configura un estilo limpio y académico (estilo APA/USAC) para todas las figuras."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "font.family": "DejaVu Sans",
        }
    )


def plot_correlation_matrix(
    corr_df: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 10),
) -> plt.Figure:
    """Grafica matriz de correlación de Pearson triangular inferior."""
    set_thesis_style()
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=len(corr_df) <= 24,
        fmt=".2f",
        annot_kws={"size": 7 if len(corr_df) > 14 else 9},
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Correlación de Pearson (r)"},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_scree_and_variance(
    eigenvalues: Sequence[float],
    var_ratio: Sequence[float],
    title: str,
    kaiser_threshold: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grafica Scree Plot (criterio de Kaiser) + varianza explicada acumulada."""
    set_thesis_style()
    n = len(eigenvalues)
    pcs = np.arange(1, n + 1)
    cum_var = np.cumsum(var_ratio) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Scree plot
    colors = ["#1b7837" if ev >= kaiser_threshold else "#9ecae1" for ev in eigenvalues]
    ax1.bar(pcs, eigenvalues, color=colors, edgecolor="#333333", alpha=0.9)
    ax1.plot(pcs, eigenvalues, "o-", color="#222222", linewidth=1.2, markersize=4)
    ax1.axhline(
        kaiser_threshold,
        color="#d7191c",
        linestyle="--",
        linewidth=1.3,
        label=f"Criterio de Kaiser (λ = {kaiser_threshold:.1f})",
    )
    ax1.set_xticks(pcs)
    ax1.set_xticklabels([f"PC{i}" for i in pcs], rotation=45, fontsize=8)
    ax1.set_ylabel("Autovalor (λ)")
    ax1.set_xlabel("Componente Principal")
    ax1.set_title("Autovalores (Scree Plot)", fontweight="bold")
    ax1.legend(frameon=True)

    # Varianza acumulada
    ax2.bar(pcs, np.array(var_ratio) * 100.0, color="#4393c3", alpha=0.75, label="Individual (%)")
    ax2.plot(pcs, cum_var, "s-", color="#b2182b", linewidth=1.8, markersize=5, label="Acumulada (%)")
    ax2.axhline(85.0, color="#666666", linestyle=":", linewidth=1.2, label="Umbral 85%")
    ax2.set_xticks(pcs)
    ax2.set_xticklabels([f"PC{i}" for i in pcs], rotation=45, fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Varianza explicada (%)")
    ax2.set_xlabel("Componente Principal")
    ax2.set_title("Varianza Individual y Acumulada", fontweight="bold")
    ax2.legend(frameon=True, loc="center right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_band_contributions(
    contrib_df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str = "Contribución relativa en PCA retenido (%)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Barras horizontales de contribución por banda coloreadas por sensor (Óptico vs SAR)."""
    set_thesis_style()
    df_sorted = contrib_df.sort_values(value_col, ascending=True).copy()

    def _color_band(name: str) -> str:
        if name.startswith("VV") or name.startswith("VH"):
            return "#d95f02"  # Naranja SAR
        return "#1b9e77"  # Verde óptico S2

    colors = [_color_band(b) for b in df_sorted.index]
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.36 * len(df_sorted))))
    bars = ax.barh(df_sorted.index, df_sorted[value_col], color=colors, edgecolor="#333333")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.12, bar.get_y() + bar.get_height() / 2.0, f"{w:.2f}%", va="center", fontsize=8.5)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#1b9e77", edgecolor="#333333", label="Sentinel-2 (Óptico)"),
        Patch(facecolor="#d95f02", edgecolor="#333333", label="Sentinel-1 (SAR)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, df_sorted[value_col].max() * 1.18)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    title: str,
    is_proportion: bool = False,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grafica matriz de confusión (conteos o proporciones de área Olofsson)."""
    set_thesis_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    fmt = ".4f" if is_proportion else "d"
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=fmt,
        cmap="YlGnBu",
        linewidths=0.6,
        cbar_kws={"label": "Proporción de área (p_ij)" if is_proportion else "Número de muestras"},
        ax=ax,
    )
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel(cm_df.columns.name or "Referencia")
    ax.set_ylabel(cm_df.index.name or "Predicho")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_model_comparison(
    summary_df: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Gráfico comparativo de OA, Kappa y F1 Manglar con barras de error (IC 95%)."""
    set_thesis_style()
    models = summary_df["Modelo"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.4))

    metrics_spec = [
        ("OA", "OA_CI_half", "Precisión Global (OA)", "#2b83ba", -width),
        ("Kappa", "Kappa_CI_half", "Índice Kappa (κ)", "#abdda4", 0.0),
        ("F1_Mangle", "F1_Mangle_CI_half", "F1-Score Manglar", "#006837", width),
    ]

    for val_col, err_col, label, color, offset in metrics_spec:
        vals = summary_df[val_col].values * 100.0
        errs = (
            summary_df[err_col].values * 100.0
            if err_col in summary_df.columns and not summary_df[err_col].isna().all()
            else None
        )
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            label=label,
            color=color,
            edgecolor="#222222",
            alpha=0.92,
        )
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + (2.5 if errs is not None else 1.2),
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Porcentaje (%) / Kappa × 100")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig

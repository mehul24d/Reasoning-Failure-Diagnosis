"""Static report figures (matplotlib, PNG).

Palette: categorical slots 1-3 of the dataviz reference palette in fixed order
(blue, orange, aqua); controls / random-null bands in neutral grey; text in
neutral ink, never series colour; recessive grid; one y-axis per panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
NULL = "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")


def _save(fig: Any, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _end_label(ax: Any, x: np.ndarray, y: np.ndarray, text: str) -> None:
    finite = np.flatnonzero(np.isfinite(y))
    if len(finite):
        i = finite[-1]
        ax.annotate(
            text,
            (x[i], y[i]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK2,
            va="center",
        )


# --------------------------------------------------------------------------- baseline A


def baseline_a_figures(
    summary: dict[str, Any], out: Path, n_layers: int, places: list[int]
) -> None:
    layers = np.arange(n_layers)
    names = {0: "units", 1: "tens", 2: "hundreds", 3: "thousands"}
    counts = summary.get("error_place_counts", {})
    show = sorted(sorted(places, key=lambda p: -counts.get(p, counts.get(str(p), 0)))[:3])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, p in enumerate(show):
        y = np.array(summary["curves"]["per_place_test_bacc"][p], float)
        # curves converge in late layers, so end labels would collide: legend only
        ax.plot(
            layers,
            y,
            color=SERIES[k],
            linewidth=1.6,
            label=f"place {p} ({names.get(p, f'10^{p}')})",
        )
    ax.axhline(0.1, color=NULL, linestyle=":", linewidth=1.2, label="chance (10 digits)")
    ax.set_ylim(0, 1.02)
    _style(
        ax,
        "Gold-digit decodability on held-out correct cases\n(3 places with most errors)",
        "layer",
        "balanced accuracy",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _save(fig, out / "fig_a_decodability_by_layer.png")

    rec = summary["curves"]["err_recovery_by_layer"]
    real = np.array(rec["err_pred"], float)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(layers, real[:, 0], color=SERIES[0], linewidth=1.6, label="probe = correct digit")
    ax.plot(layers, real[:, 1], color=SERIES[1], linewidth=1.6, label="probe = model's wrong digit")
    ax.plot(
        layers,
        np.array(rec["err_pred_shuf"], float)[:, 0],
        color=NULL,
        linestyle="--",
        linewidth=1.2,
        label="shuffled-label probe = correct",
    )
    ax.plot(
        layers,
        np.array(rec["err_pred_rand"], float)[:, 0],
        color=NULL,
        linestyle=":",
        linewidth=1.2,
        label="random probe = correct",
    )
    ax.set_ylim(0, 1.02)
    _style(
        ax,
        f"Error cases (n={summary['primary_P1_correct_only']['n_error_items']}): "
        "what the correct-only probe reads at the wrong place",
        "layer",
        "fraction of error cases",
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _save(fig, out / "fig_a_error_recovery_by_layer.png")


# --------------------------------------------------------------------------- baseline B


def auroc_by_layer(
    curves: dict[str, list[float]], shuffled: list[float] | None, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (name, ys) in enumerate(curves.items()):
        y = np.array(ys, float)
        x = np.arange(len(y))
        ax.plot(x, y, color=SERIES[k], linewidth=1.6, label=name)
        _end_label(ax, x, y, name)
    if shuffled is not None:
        ax.plot(
            np.arange(len(shuffled)),
            shuffled,
            color=NULL,
            linestyle="--",
            linewidth=1.2,
            label="shuffled labels",
        )
    ax.axhline(0.5, color=NULL, linestyle=":", linewidth=1.0)
    ax.set_ylim(0.3, 1.0)
    _style(ax, title, "layer", "train-CV AUROC (template-grouped)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _save(fig, path)


def strength_sweep(sweep: dict[str, dict[str, Any]], path: Path, title: str) -> None:
    """sweep[direction] = {"strengths": [...], "net": [...], "repair": [...], "damage": [...],
    "null_lo": [...], "null_hi": [...]} (null band optional)."""
    dirs = list(sweep)
    fig, axes = plt.subplots(
        1, len(dirs), figsize=(3.4 * len(dirs), 3.4), sharey=True, squeeze=False
    )
    for ax, name in zip(axes[0], dirs):
        s = sweep[name]
        x = np.array(s["strengths"], float)
        if s.get("null_lo") is not None:
            ax.fill_between(
                x,
                s["null_lo"],
                s["null_hi"],
                color=NULL,
                alpha=0.25,
                linewidth=0,
                label="random dirs (2.5-97.5%)",
            )
        ax.plot(
            x,
            np.array(s["repair"]) * 100,
            color=SERIES[0],
            linewidth=1.6,
            marker="o",
            ms=4,
            label="repair %",
        )
        ax.plot(
            x,
            np.array(s["damage"]) * 100,
            color=SERIES[1],
            linewidth=1.6,
            marker="o",
            ms=4,
            label="damage %",
        )
        ax.plot(
            x,
            np.array(s["net"]) * 100,
            color=SERIES[2],
            linewidth=1.6,
            marker="o",
            ms=4,
            label="net gain (pts)",
        )
        ax.axhline(0, color="#c9c8c2", linewidth=0.8)
        if np.all(x > 0) and x.max() / max(x.min(), 1e-9) > 8:
            ax.set_xscale("log")
        _style(ax, name, "strength", "percent / points")
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    _save(fig, path)


# --------------------------------------------------------------------------- pilot


def readability_maps(maps: dict[str, np.ndarray], path: Path, col_labels: list[str]) -> None:
    """maps[variable] = [n_layers, n_steps] metric (AUROC or balanced acc)."""
    names = list(maps)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        m = np.asarray(maps[name], float)
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="Blues", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(col_labels[: m.shape[1]], fontsize=7)
        _style(ax, name, "step position", "layer")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7, colors=INK2)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    _save(fig, path)


def condition_bars(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    """rows: {"name", "net", "lo", "hi"} (points); optional null band in the row named 'random'."""
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        color = NULL if r["name"].startswith("random") else SERIES[0]
        ax.barh(yi, r["net"], color=color, height=0.55)
        if r.get("lo") is not None:
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=INK, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8, color=INK2)
    ax.axvline(0, color="#c9c8c2", linewidth=0.8)
    _style(ax, title, "net gain (accuracy points)", "")
    _save(fig, path)

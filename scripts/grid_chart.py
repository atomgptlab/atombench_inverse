#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from jarvis.io.vasp.inputs import Poscar

# ───────────────────────── style ─────────────────────────
mpl.rcParams["font.family"] = "serif"
WVU_BLUE = "#002855"  # target
WVU_GOLD = "#EEAA00"  # predicted (WVU gold)

MODELS = ["agpt", "cdvae", "flowmm"]
MODEL_LABEL = {"agpt": "AtomGPT", "cdvae": "CDVAE", "flowmm": "FlowMM"}

PARAMS = ["a", "c", "gamma"]
PARAM_LABEL = {"a": r"$a$ (Å)", "c": r"$c$ (Å)", "gamma": r"$\gamma$ (°)"}

# Bins similar to your earlier plots/figures
BINS = {
    "a": np.arange(2.0, 7.0, 0.1),
    "c": np.arange(2.0, 7.0, 0.1),
    "gamma": np.arange(30.0, 150.0, 10.0),
}

# ───────────────────── helpers: discovery ─────────────────────
def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Newest CSV under dir_path (recursively) that has id/target/prediction columns."""
    latest: Optional[Tuple[float, Path]] = None
    for p, _, files in os.walk(dir_path):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = Path(p) / f
            try:
                with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    fields = [fn.strip().lower() for fn in (reader.fieldnames or [])]
                    if {"id", "target", "prediction"}.issubset(set(fields)):
                        mt = path.stat().st_mtime
                        if latest is None or mt > latest[0]:
                            latest = (mt, path)
            except Exception:
                continue
    return latest[1] if latest else None


def discover_model_dirs(root: Path, tag: str) -> Dict[str, Path]:
    """
    From job_runs root, find dirs for each model for a dataset tag ('alex' or 'jarvis').
    Returns a subset of {'agpt': Path, 'cdvae': Path, 'flowmm': Path}.
    """
    out: Dict[str, Path] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.lower()
        if tag not in name:
            continue
        for m in MODELS:
            if m in name:
                out[m] = entry
                break
    return out

# ───────────────────── helpers: parsing ─────────────────────
def _unescape_poscar(s: str) -> str:
    return (
        s.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\\n", "\n")
        .replace("\\t", " ")
        .strip()
    )


def extract_series(csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Read a benchmark CSV -> { 'target': {'a','c','gamma'}, 'predicted': {…} } as lists of floats.
    """
    out = {"target": {k: [] for k in PARAMS}, "predicted": {k: [] for k in PARAMS}}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tgt = Poscar.from_string(_unescape_poscar(row["target"])).atoms.to_dict()
                pred = Poscar.from_string(_unescape_poscar(row["prediction"])).atoms.to_dict()

                out["target"]["a"].append(float(tgt["abc"][0]))
                out["target"]["c"].append(float(tgt["abc"][2]))
                out["target"]["gamma"].append(float(tgt["angles"][2]))

                out["predicted"]["a"].append(float(pred["abc"][0]))
                out["predicted"]["c"].append(float(pred["abc"][2]))
                out["predicted"]["gamma"].append(float(pred["angles"][2]))
            except Exception:
                # skip bad rows
                continue
    return out


def load_klds(metrics_path: Path) -> Dict[str, float]:
    """
    Load KLD values from metrics.json -> {'a':…, 'c':…, 'gamma':…}.
    Missing keys return {} (we'll just omit annotation).
    """
    try:
        data = json.loads(metrics_path.read_text())
        k = data.get("KLD", {})
        return {
            "a": float(k.get("a")),
            "c": float(k.get("c")),
            "gamma": float(k.get("gamma")),
        }
    except Exception:
        return {}

# ───────────────────── helpers: plotting ─────────────────────
def _weights_percent(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) * (100.0 / n) if n > 0 else np.array([])


def annotate_kld(ax: plt.Axes, value: Optional[float]) -> None:
    if value is None:
        return
    ax.text(
        0.97,
        0.92,
        f"KLD = {value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
    )


def plot_dataset_grid(tag: str, root: Path, out_png: Path, title: str) -> None:
    """
    Build a 3×3 grid for one dataset tag ('alex' or 'jarvis'):
      rows = models (AtomGPT, CDVAE, FlowMM)
      cols = parameters (a, c, gamma)
    """
    model_dirs = discover_model_dirs(root, tag)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.5), constrained_layout=True)

    for r, model in enumerate(MODELS):
        axrow = axes[r]
        label = MODEL_LABEL[model]
        bench_dir = model_dirs.get(model)

        series = None
        klds = {}
        if bench_dir:
            csv_path = find_benchmark_csv(bench_dir)
            if csv_path:
                series = extract_series(csv_path)
            mfp = bench_dir / "metrics.json"
            if mfp.exists():
                klds = load_klds(mfp)

        for c, param in enumerate(PARAMS):
            ax = axrow[c]
            if series is None or not series["target"][param] or not series["predicted"][param]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            xt = np.asarray(series["target"][param], dtype=float)
            xp = np.asarray(series["predicted"][param], dtype=float)

            bins = BINS[param]
            wt_t = _weights_percent(len(xt))
            wt_p = _weights_percent(len(xp))

            # target (blue) then predicted (gold)
            ax.hist(xt, bins=bins, weights=wt_t, alpha=0.70, color=WVU_BLUE, label="target")
            ax.hist(xp, bins=bins, weights=wt_p, alpha=0.70, color=WVU_GOLD, label="predicted")

            # Titles/labels layout
            if r == 0:
                ax.set_title(PARAM_LABEL[param], fontsize=16)
            if c == 0:
                ax.set_ylabel(label, fontsize=14)
            else:
                ax.set_yticklabels([])  # reduce clutter
            if r == 2:
                ax.set_xlabel(PARAM_LABEL[param], fontsize=12)
            else:
                ax.set_xlabel("")

            # KLD annotation from metrics.json
            annotate_kld(ax, klds.get(param))

    # One legend in the top-left panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=True, fontsize=10)

    # Add a big y-axis label on the right like your paper figure
    fig.suptitle(title, fontsize=20)
    # optional: global y label; uses a figure-level text at right
    fig.text(0.99, 0.5, "Materials Percentage (%)", rotation=90,
             va="center", ha="right", fontsize=12)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"✓ wrote {out_png}")


# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct 3×3 overlay grids for Alexandria and JARVIS; annotate each panel with KLD from metrics.json."
    )
    ap.add_argument("--root", type=Path, default=Path("job_runs"), help="Root containing benchmark subdirs")
    ap.add_argument("--outdir", type=Path, default=Path("figures"), help="Directory to write PNGs")
    ap.add_argument("--alex-title", default="Alexandria DS-A/B", help="Suptitle for Alexandria figure")
    ap.add_argument("--jarvis-title", default="JARVIS Supercon-3D", help="Suptitle for JARVIS figure")
    args = ap.parse_args()

    plot_dataset_grid("alex", args.root, args.outdir / "alexandria_reconstruction_grid.png", args.alex_title)
    plot_dataset_grid("jarvis", args.root, args.outdir / "jarvis_reconstruction_grid.png", args.jarvis_title)


if __name__ == "__main__":
    main()


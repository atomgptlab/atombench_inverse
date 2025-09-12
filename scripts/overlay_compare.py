#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Overlay Tc & Crystal-System Comparisons (JARVIS vs Alexandria)
# - Normalized to percent for fair comparison
# - Crystal systems via pymatgen SpacegroupAnalyzer.get_crystal_system()
# - Caches pre-plot arrays (Tc lists & crystal-system lists) for fast re-render
###############################################################################

import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.figshare import data as jarvis_data

# ── Matplotlib defaults (match your example) ───────────────────────────────
mpl.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 18})

CRYSYS_ORDER = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]
CRYSYS_LABELS = [s.capitalize() for s in CRYSYS_ORDER]


# ── Helpers ────────────────────────────────────────────────────────────────
def _valid_float(x) -> Optional[float]:
    try:
        f = float(x)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _get_crystal_system(pmg: Structure, symprec: float = 0.1) -> Optional[str]:
    """Return crystal system string via SpacegroupAnalyzer.get_crystal_system(),
    computed on the conventional standardized structure for robustness.
    """
    try:
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        # normalize to lowercase for consistent counting
        return cs.lower() if isinstance(cs, str) else None
    except Exception:
        return None


def _weights_percent(n: int) -> np.ndarray:
    """Histogram weights so bars reflect percent of dataset."""
    if n <= 0:
        return np.array([])
    return np.ones(n, dtype=float) * (100.0 / n)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── Data collection (Alexandria CSV) ───────────────────────────────────────
def collect_alexandria(
    csv_files: List[Path],
    target_key: str,
    struct_key: str,
    symprec: float,
    max_size: Optional[int],
) -> Dict[str, List]:
    """Return {'tc': [...], 'crystal_systems': [...]}."""
    dfs = [pd.read_csv(p) for p in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    tc_vals: List[float] = []
    crysys: List[str] = []

    for row in tqdm(
        df_all.itertuples(index=False),
        total=len(df_all),
        desc="Parsing Alexandria CSVs",
    ):
        if max_size is not None and len(tc_vals) >= max_size:
            break

        # Structure
        try:
            pmg = Structure.from_dict(ast.literal_eval(getattr(row, struct_key)))
        except Exception:
            continue

        # Tc
        tgt = getattr(row, target_key, None)
        tcf = _valid_float(tgt)
        if tcf is not None:
            tc_vals.append(tcf)

        # Crystal system
        cs = _get_crystal_system(pmg, symprec=symprec)
        if cs in CRYSYS_ORDER:
            crysys.append(cs)

    return {"tc": tc_vals, "crystal_systems": crysys}


# ── Data collection (JARVIS figshare) ─────────────────────────────────────
def collect_jarvis(
    dataset_name: str,
    target_key: str,
    symprec: float,
    max_size: Optional[int],
) -> Dict[str, List]:
    """Return {'tc': [...], 'crystal_systems': [...]} from figshare dataset."""
    tc_vals: List[float] = []
    crysys: List[str] = []

    for item in tqdm(jarvis_data(dataset_name), desc="Downloading JARVIS"):
        if max_size is not None and len(tc_vals) >= max_size:
            break

        # Structure
        try:
            atoms_j = JarvisAtoms.from_dict(item["atoms"])
            pmg: Structure = atoms_j.pymatgen_converter()
        except Exception:
            continue

        # Tc
        tcf = _valid_float(item.get(target_key, None))
        if tcf is not None:
            tc_vals.append(tcf)

        # Crystal system
        cs = _get_crystal_system(pmg, symprec=symprec)
        if cs in CRYSYS_ORDER:
            crysys.append(cs)

    return {"tc": tc_vals, "crystal_systems": crysys}


# ── Plotting ───────────────────────────────────────────────────────────────
def plot_tc_overlay(
    tc_a: List[float],
    tc_j: List[float],
    out_png: Path,
    tc_min: float,
    tc_max: float,
    tc_step: float,
) -> None:
    bins = np.arange(tc_min, tc_max + tc_step, tc_step)

    fig, ax = plt.subplots(figsize=(11.5, 11.5), constrained_layout=True)

    w_a = _weights_percent(len(tc_a))
    w_j = _weights_percent(len(tc_j))

    # Alexandria (plum), JARVIS (tab:blue) — match your overlay look
    ax.hist(
        tc_j,
        bins=bins,
        weights=w_j,
        alpha=0.6,
        label="JARVIS Supercon-3D",
        color="#002855",
    )
    ax.hist(
        tc_a,
        bins=bins,
        weights=w_a,
        alpha=0.6,
        label="Alexandria DS-A/DS-B",
        color="#EEAA00",
    )

    ax.set_xlabel(r"$T_c$ (K)", fontsize=30)
    ax.set_ylabel("Materials dist. (%)", fontsize=30)
    ax.set_title("Distribution of $T_c$\nJARVIS vs Alexandria", fontsize=38)
    ax.set_xticks(np.linspace(tc_min, tc_max, 10))
    ax.tick_params(axis="x", labelsize=23)
    plt.yticks(fontsize=23)
    # ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=20)

    _ensure_dir(out_png.parent)
    plt.savefig(out_png, format="png", dpi=200)
    plt.close(fig)


def _crysys_percent_hist(cs_list: List[str]) -> np.ndarray:
    """Return a length-7 array of percentages in CRYSYS_ORDER."""
    if not cs_list:
        return np.zeros(len(CRYSYS_ORDER))
    counts = pd.Series(cs_list).value_counts()
    total = counts.sum()
    percents = [(counts.get(cs, 0) / total) * 100.0 for cs in CRYSYS_ORDER]
    return np.array(percents, dtype=float)


def plot_crysys_overlay(
    cs_a: List[str],
    cs_j: List[str],
    out_png: Path,
) -> None:
    # Compute percent per system
    p_a = _crysys_percent_hist(cs_a)
    p_j = _crysys_percent_hist(cs_j)

    x = np.arange(len(CRYSYS_ORDER))

    fig, ax = plt.subplots(figsize=(11.5, 11.5), constrained_layout=True)

    # Overlay bars at identical positions (alpha) to mimic your style
    bar_w = 0.6
    ax.bar(
        x,
        p_j,
        width=bar_w,
        alpha=0.6,
        label="JARVIS Supercon-3D",
        color="#002855",
    )
    ax.bar(
        x,
        p_a,
        width=bar_w,
        alpha=0.6,
        label="Alexandria DS-A/DS-B",
        color="#EEAA00",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CRYSYS_LABELS, rotation=25)
    ax.set_xlabel("Crystal system", fontsize=30)
    ax.set_ylabel("% of Total Structures", fontsize=30)
    ax.set_title("Crystal Systems \n JARVIS vs Alexandria", fontsize=38)
    # ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=23)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    _ensure_dir(out_png.parent)
    plt.savefig(out_png, format="png", dpi=200)
    plt.close(fig)


# ── CLI & main ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overlay_compare.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Generate overlaid, normalized comparisons between JARVIS and Alexandria:\n"
            "  1) tc_overlay_compare.png  (normalized Tc histogram overlay)\n"
            "  2) crystal_system_overlay_compare.png  "
            "(normalized crystal-system overlay)\n\n"
            "Caching: stores pre-plot arrays (Tc values, crystal systems) so you can\n"
            "re-render instantly after tweaking visual settings."
        ),
    )
    # Alexandria CSV inputs
    p.add_argument(
        "--alex-csv-files",
        nargs="+",
        required=True,
        help="Alexandria CSV file paths",
    )
    p.add_argument(
        "--alex-target-key",
        default="Tc",
        help="Alexandria target column (Tc)",
    )
    p.add_argument(
        "--alex-struct-key",
        default="structure",
        help="Alexandria column with pymatgen Structure dict",
    )

    # JARVIS inputs
    p.add_argument(
        "--jarvis-dataset",
        required=True,
        help="JARVIS figshare dataset name (e.g., 'dft_3d')",
    )
    p.add_argument(
        "--jarvis-target-key",
        default="Tc_supercon",
        help="JARVIS Tc column (e.g., Tc_supercon)",
    )

    # Symmetry / performance
    p.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Symmetry tolerance for SpacegroupAnalyzer",
    )
    p.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional cap per dataset for quick tests",
    )

    # Binning / aesthetics
    p.add_argument("--tc-min", type=float, default=0.0)
    p.add_argument("--tc-max", type=float, default=45.0)
    p.add_argument(
        "--tc-step",
        type=float,
        default=1.5,
        help="Step size for identical Tc bin edges",
    )

    # Output & caching
    p.add_argument(
        "--output",
        required=True,
        help="Output directory for PNGs and cache",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild cache (recompute symmetry)",
    )

    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)

    out_dir = Path(args.output).expanduser().resolve()
    cache_dir = out_dir / "cache"
    _ensure_dir(out_dir)
    _ensure_dir(cache_dir)

    cache_path = cache_dir / "preplot_payload.json"

    if args.refresh or (not cache_path.exists()):
        # Build pre-plot payload (heavy part)
        alex = collect_alexandria(
            csv_files=[Path(p).expanduser().resolve() for p in args.alex_csv_files],
            target_key=args.alex_target_key,
            struct_key=args.alex_struct_key,
            symprec=args.symprec,
            max_size=args.max_size,
        )
        jarv = collect_jarvis(
            dataset_name=args.jarvis_dataset,
            target_key=args.jarvis_target_key,
            symprec=args.symprec,
            max_size=args.max_size,
        )

        payload = {
            "alexandria": {
                "n_tc": len(alex["tc"]),
                "n_crysys": len(alex["crystal_systems"]),
                "tc": alex["tc"],
                "crystal_systems": alex["crystal_systems"],
            },
            "jarvis": {
                "n_tc": len(jarv["tc"]),
                "n_crysys": len(jarv["crystal_systems"]),
                "tc": jarv["tc"],
                "crystal_systems": jarv["crystal_systems"],
            },
            "meta": {
                "symprec": args.symprec,
                "source_note": (
                    "Tc values use provided columns; crystal systems via "
                    "SpacegroupAnalyzer.get_crystal_system() on conventional "
                    "standardized structures."
                ),
            },
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f)
        print(f"✓ Cached pre-plot payload → {cache_path}")
    else:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        print(f"✓ Loaded cache from {cache_path}")

    # Extract arrays (lightweight plotting)
    tc_a = payload["alexandria"]["tc"]
    tc_j = payload["jarvis"]["tc"]
    cs_a = payload["alexandria"]["crystal_systems"]
    cs_j = payload["jarvis"]["crystal_systems"]

    # PNG 1: Tc overlay (normalized)
    plot_tc_overlay(
        tc_a=tc_a,
        tc_j=tc_j,
        out_png=out_dir / "tc_overlay_compare.png",
        tc_min=args.tc_min,
        tc_max=args.tc_max,
        tc_step=args.tc_step,
    )

    # PNG 2: Crystal-system overlay (normalized)
    plot_crysys_overlay(
        cs_a=cs_a,
        cs_j=cs_j,
        out_png=out_dir / "crystal_system_overlay_compare.png",
    )

    # Also drop CSV summaries for reproducibility
    counts_dir = out_dir / "counts"
    _ensure_dir(counts_dir)
    # crystal system percent tables
    pd.DataFrame(
        {
            "crystal_system": CRYSYS_LABELS,
            "jarvis_percent": _crysys_percent_hist(cs_j),
            "alexandria_percent": _crysys_percent_hist(cs_a),
        }
    ).to_csv(counts_dir / "crystal_system_percentages.csv", index=False)

    print(f"✓ Wrote {out_dir / 'tc_overlay_compare.png'}")
    print(f"✓ Wrote {out_dir / 'crystal_system_overlay_compare.png'}")
    print(f"✓ Wrote {counts_dir / 'crystal_system_percentages.csv'}")


if __name__ == "__main__":
    main()

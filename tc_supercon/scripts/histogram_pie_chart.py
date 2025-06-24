#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Imports
###############################################################################
import argparse
import hashlib
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from tqdm import tqdm

###############################################################################
# General helpers
###############################################################################
def deterministic_split(
    n: int, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_val = int(val_ratio * n)
    n_test = int(test_ratio * n)
    n_train = n - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def hash10(strings: List[str]) -> str:
    h = hashlib.sha256()
    for s in strings:
        h.update(s.encode())
        h.update(b",")
    return h.hexdigest()[:10]


def canonicalise(pmg: Structure, symprec: float = 0.1):
    """Return (cif_raw, cif_conv, spg_raw, spg_conv). Always succeeds."""
    try:
        cif_raw = pmg.to(fmt="cif")
    except Exception:
        cif_raw = ""
    try:
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        spg_raw = sga.get_space_group_number()
        conv = sga.get_conventional_standard_structure()
        cif_conv = conv.to(fmt="cif")
        spg_conv = SpacegroupAnalyzer(conv, symprec=symprec).get_space_group_number()
    except Exception:
        cif_conv, spg_raw, spg_conv = "", -1, -1
    return cif_raw, cif_conv, spg_raw, spg_conv

###############################################################################
# Dataset collection
###############################################################################
def collect_records(
    dataset_name: str,
    id_key: str,
    target_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    """Download dataset, keep those with a valid *target_key*.

    Returned DataFrame columns:
        material_id, atoms_j (jarvis Atoms), cif_raw, cif_conv, spg_raw,
        spg_conv, pretty_formula, elements (list[str]), target_key
    """
    records: List[Dict] = []
    for item in tqdm(jarvis_data(dataset_name), desc="Downloading/JARVIS"):
        tgt = item.get(target_key, "na")
        if tgt in ("na", None):
            continue

        # Build structural objects
        atoms_j = Atoms.from_dict(item["atoms"])
        pmg: Structure = atoms_j.pymatgen_converter()

        try:
            cif_raw, cif_conv, spg_raw, spg_conv = canonicalise(pmg)
            if not cif_raw:  # canonicalise failed badly
                continue
        except Exception:
            continue

        records.append(
            {
                "material_id": item[id_key],
                "atoms_j": atoms_j,
                "cif_raw": cif_raw,
                "cif_conv": cif_conv,
                "spg_raw": spg_raw,
                "spg_conv": spg_conv,
                "pretty_formula": pmg.composition.reduced_formula,
                "elements": [el.symbol for el in pmg.species],
                target_key: tgt,
            }
        )

        if max_size is not None and len(records) == max_size:
            break

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            f"No usable entries found for dataset '{dataset_name}' with target "
            f"'{target_key}'."
        )
    return df

###############################################################################
# Tc Histogram
###############################################################################
def create_tc_histogram(
    df: pd.DataFrame, target_key: str, output_dir: Path
) -> None:
    temps = df[target_key]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        temps,
        bins=30,
        density=False,
        cumulative=False,
        alpha=0.6,
        label="Distribution of Tc in Alexandria Dataset",
    )

    ax.set_xlabel("Tc")
    ax.set_ylabel("Number of Structures")
    ax.legend()

    (output_dir / "alex_tc_histogram.pdf").with_suffix(".pdf").parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(
        output_dir / "alex_tc_histogram.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

###############################################################################
# Composition Pie Chart
###############################################################################
def create_composition_pie_chart(df: pd.DataFrame, output_dir: Path) -> None:
    element_counts = df["elements"].explode().value_counts()
    top25 = element_counts.iloc[:25].copy()
    others = element_counts.iloc[25:].sum()
    if others:
        top25.loc["Other"] = others
    (output_dir / "element_counts.csv").parent.mkdir(parents=True, exist_ok=True)
    top25.to_csv(output_dir / "element_counts.csv", header=["count"])
    counts = top25.values
    labels = top25.index.tolist()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts,
        labels=labels,           # ← length now matches counts
        labeldistance=1.3,
	radius=0.8,
	shadow=False,
        startangle=90,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
        textprops={"fontsize": 12},
    )
    ax.axis("equal")
    plt.title("Number of Elements in the Alexandria Tc Dataset")

    (output_dir / "alex_composition_pie_chart.pdf").parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(
        output_dir / "alex_composition_pie_chart.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

###############################################################################
# CLI handling
###############################################################################
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv-files", nargs="+", required=True, help="Input CSV paths")
    common.add_argument("--id-key", default="mat_id", help="Column with unique IDs")
    common.add_argument("--target", dest="target_key", 
                        default="Tc", help="Target column")
    common.add_argument("--structure-key", dest="struct_key", default="structure",
                        help="Column that stores the pymatgen Structure dict")
    common.add_argument("--output", required=True, help="Output directory")
    common.add_argument("--max-size", type=int, default=None)
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--val-ratio", type=float, default=0.1)
    common.add_argument("--test-ratio", type=float, default=0.1)

    return argparse.ArgumentParser(
        prog="histogram_pie_chart.py",
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            create Tc histogram and elemental composition pie chart
            from JARVIS tc_supercon dft_3d
            """
        ),
    )

###############################################################################
# Main
###############################################################################
def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)

    csv_files = [Path(p).expanduser().resolve() for p in args.csv_files]
    print(f"CSV files: {', '.join(str(p) for p in csv_files)}")

    # 1) Collect dataset
    df = collect_records(
        dataset_name=args.dataset,
        id_key=args.id_key,
        target_key=args.target_key,
        max_size=args.max_size,
    )
    n_samples = len(df)
    print(f"✓ Collected {n_samples} usable entries")
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Create Tc histogram
    create_tc_histogram(df, args.target_key, out_dir)

    # 3) Create composition pie chart
    create_composition_pie_chart(df, out_dir)


if __name__ == "__main__":
    main()

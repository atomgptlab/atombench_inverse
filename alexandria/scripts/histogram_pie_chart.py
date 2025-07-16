#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Imports
###############################################################################

import os
import argparse
import ast
import hashlib
import random
import textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.core.atoms import pmg_to_atoms, Atoms
from tqdm import tqdm

mpl.rcParams['font.family']    = 'serif'

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
def collect_records_from_csv(
    csv_files: List[Path],
    id_key: str,
    target_key: str,
    struct_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    records: List[Dict] = []
    for row in tqdm(df_all.itertuples(
        index=False), total=len(df_all), desc="Parsing CSVs"):
        if max_size is not None and len(records) >= max_size:
            break

        tgt = getattr(row, target_key, "na")
        if tgt in (None, "na") or (isinstance(tgt, float) and np.isnan(tgt)):
            continue

        try:
            pmg_struct = Structure.from_dict(ast.literal_eval(getattr(row, struct_key)))
        except Exception:
            continue

        try:
            atoms = pmg_to_atoms(pmg_struct)
        except Exception:
            # fallback manual conversion
            atoms = Atoms(
                lattice=pmg_struct.lattice.matrix.tolist(),
                elements=[str(s.specie) for s in pmg_struct],
                coords=pmg_struct.cart_coords.tolist(),
                coords_are_cartesian=True,
            )

        cif_raw, cif_conv, spg_raw, spg_conv = canonicalise(pmg_struct)

        records.append(
            {
                "material_id": getattr(row, id_key),
                "atoms": atoms,
                "cif_raw": cif_raw,
                "cif_conv": cif_conv,
                "spg_raw": spg_raw,
                "spg_conv": spg_conv,
                "pretty_formula": pmg_struct.composition.reduced_formula,
                "elements": [el.symbol for el in pmg_struct.species],
                target_key: tgt,
            }
        )

    if not records:
        raise RuntimeError(
            "No valid entries collected – check column names or the max-size filter."
        )

    return pd.DataFrame(records)

###############################################################################
# Tc Histogram
###############################################################################
def create_tc_histogram(
    df: pd.DataFrame, target_key: str, output_dir: Path
) -> None:
    temps = df[target_key]

    fig, ax = plt.subplots(
            figsize=(11.5, 11.5),
            constrained_layout=True
    )
    ax.hist(
        temps,
        bins=30,
        density=False,
        cumulative=False,
        alpha=1,
    )

    ax.set_xlabel("$T_{c}$ (K)", fontsize=30)
    ax.set_ylabel("Number of Structures", fontsize=30)

    (output_dir / "alex_tc_histogram.png").with_suffix(".png").parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.title(
        "Distribution of $T_{c}$ in the \n Alexandria DS-A / DS-B Dataset",
        fontsize=38
    )
    plt.xticks(ticks=np.linspace(0,45,10), fontsize=23)
    plt.yticks(fontsize=23)
    plt.savefig(
        output_dir / "alex_tc_histogram.png",
        format="png"
        #bbox_inches="tight",
    )
    plt.close(fig)
###############################################################################
# Composition Pie Chart
###############################################################################
def create_composition_pie_chart(df: pd.DataFrame, output_dir: Path) -> None:
    element_counts = df["elements"].explode().value_counts()
    top_num = 23
    top = element_counts.iloc[:top_num].copy()
    others = element_counts.iloc[top_num:].sum()
    if others:
        top.loc["Other"] = others
    (output_dir / "alex_element_counts.csv").parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(output_dir / "alex_element_counts.csv", header=["count"])
    counts = top.values
    labels = top.index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    patches, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        labeldistance=1.05,
        autopct='%1.1f%%',
        pctdistance=0.8,
        radius=1,
        shadow=False,
        startangle=90,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
    )

    for txt in texts:
        txt.set_fontsize(18)
        txt.set_fontweight("bold")
        txt.set_color("black")
        txt.set_visible(True)

    counter = 11
    for pct_txt, wedge in zip(autotexts, patches):
        if counter > 0:
            ang = 0.5 * (wedge.theta1 + wedge.theta2)

            if 90 < ang < 270:
                ang += 180

            pct_txt.set_rotation(ang)         # face the correct direction
            pct_txt.set_rotation_mode('anchor')
            pct_txt.set_ha('center')
            pct_txt.set_va('center')
            pct_txt.set_fontsize(18)          # style as you wish
            #pct_txt.set_fontweight("bold")

            counter -= 1
        else:
            pct_txt.set_visible(False)


    ax.axis("equal")
    plt.title(
            "Element Proportions in the \n Alexandria DS-A / DS-B Dataset",
            fontsize=30
    )

    (output_dir / "alex_composition_pie_chart.png").parent.mkdir(
        parents=True, exist_ok=True
    )
    plt.savefig(
        output_dir / "alex_composition_pie_chart.png",
        format="png"
        #bbox_inches="tight",
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
        prog="alexandria_preprocess",
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Pre-process Alexandria CSVs into model-specific formats while
            sharing a single deterministic train/val/test split.
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
    if not os.path.exists('./df.pkl'):
        df = collect_records_from_csv(
            csv_files,
            id_key=args.id_key,
            target_key=args.target_key,
            struct_key=args.struct_key,
            max_size=args.max_size,
        )
        df.to_pickle('./df.pkl')
    if os.path.exists('./df.pkl'):
        df = pd.read_pickle('./df.pkl')
    n = len(df)
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Collected {n} usable rows")

    # 2) Create Tc histogram
    hist_path = './alex_tc_histogram.png'
    if os.path.exists(hist_path):
        os.remove(hist_path)
    create_tc_histogram(df, args.target_key, out_dir)

    # 3) Create composition pie chart
    pie_path='./alex_composition_pie_chart.png'
    if os.path.exists(pie_path):
        os.remove(pie_path)
    create_composition_pie_chart(df, out_dir)


if __name__ == "__main__":
    main()

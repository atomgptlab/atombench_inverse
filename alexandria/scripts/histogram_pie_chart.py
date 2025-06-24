#!/usr/bin/env python3
"""
Unified preprocessing for Alexandria CSV superconductivity data → AtomGPT, CDVAE, FlowMM.

The interface mirrors the earlier JARVIS‐based script – just swap
`data_preprocess.py` for `alexandria_preprocess.py`.

Example
-------
# AtomGPT outputs in ./atomgpt_csv
python alexandria_preprocess.py atomgpt \
       --csv-files alex1.csv alex2.csv \
       --output ./atomgpt_csv \
       --max-size 2000 --seed 3407

# CDVAE outputs in ./cdvae_csv
python alexandria_preprocess.py cdvae   --csv-files alex1.csv alex2.csv --output ./cdvae_csv

# FlowMM outputs in ./flowmm_csv (same split!)
python alexandria_preprocess.py flowmm  --csv-files alex1.csv alex2.csv --output ./flowmm_csv

Options like --val-ratio, --test-ratio, --seed, --target, --id-key behave the
same across sub‑commands – use the *same* values to guarantee identical test
sets.
"""
from __future__ import annotations

import os
import argparse
import ast
import hashlib
import json
import random
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.core.atoms import pmg_to_atoms, Atoms
from jarvis.io.vasp.inputs import Poscar
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def _pushd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

################################################################################
# General helpers
################################################################################

def deterministic_split(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
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

################################################################################
# Dataset collection
################################################################################

def collect_records_from_csv(
    csv_files: List[Path],
    id_key: str,
    target_key: str,
    max_size: Optional[int],
) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_files]
    df_all = pd.concat(dfs, ignore_index=True)

    records: List[Dict] = []
    for row in tqdm(df_all.itertuples(index=False), total=len(df_all), desc="Parsing CSVs"):
        if max_size is not None and len(records) >= max_size:
            break

        tgt = getattr(row, target_key, "na")
        if tgt in (None, "na") or (isinstance(tgt, float) and np.isnan(tgt)):
            continue

        try:
            pmg_struct = Structure.from_dict(ast.literal_eval(getattr(row, "structure")))
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
        raise RuntimeError("No valid entries collected – check target column name or max‑size filter.")

    return pd.DataFrame(records)

################################################################################
# Abstract‑factory base class
################################################################################

class BaseFactory(ABC):
    def __init__(self, out_dir: Path, target_key: str):
        self.out_dir = out_dir
        self.target_key = target_key
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # common helper: write POSCARs, return list of relative paths in given order
    def _write_poscars(self, df: pd.DataFrame, indices: List[int]) -> List[str]:
        rel_paths: List[str] = []

        # ① change CWD just for this block
        with _pushd(self.out_dir):
            for idx in indices:
                row   = df.iloc[idx]
                fname = f"{row['material_id']}.vasp"

                # ② let Jarvis write; both copies now end up *here*
                Poscar(row["atoms"]).write_file(fname)

                rel_paths.append(fname)

        return rel_paths
    @abstractmethod
    def dump(self, df: pd.DataFrame, id_train: List[int], id_val: List[int], id_test: List[int]):
        ...

################################################################################
# Concrete factories
################################################################################

class AtomGPTFactory(BaseFactory):
    def dump(self, df, id_train, id_val, id_test):
        all_idx = id_test + id_train + id_val  # test first as before
        rel_paths = self._write_poscars(df, all_idx)

        id_prop = pd.DataFrame({
            "structure_path": rel_paths,
            self.target_key: [df.iloc[i][self.target_key] for i in all_idx],
        })
        id_prop.to_csv(self.out_dir / "id_prop.csv", index=False, header=False)
        print(f"✓ AtomGPT → id_prop.csv ({len(id_prop)})  hash={hash10(rel_paths)}")


class CDVAEFactory(BaseFactory):
    def dump(self, df, id_train, id_val, id_test):
        self._write_poscars(df, id_train + id_val + id_test)  # not strictly needed for CDVAE

        def _mk(idx):
            return pd.DataFrame({
                "material_id": [df.iloc[i]["material_id"] for i in idx],
                "cif": [df.iloc[i]["cif_raw"] for i in idx],
                self.target_key: [df.iloc[i][self.target_key] for i in idx],
            })

        _mk(id_train).to_csv(self.out_dir / "train.csv", index=False)
        _mk(id_val).to_csv(self.out_dir / "val.csv", index=False)
        _mk(id_test).to_csv(self.out_dir / "test.csv", index=False)
        print(f"✓ CDVAE → train/val/test CSVs (test hash={hash10([df.iloc[i]['material_id'] for i in id_test])})")


class FlowMMFactory(BaseFactory):
    def dump(self, df, id_train, id_val, id_test):
        self._write_poscars(df, id_train + id_val + id_test)

        def _mk(idx):
            return pd.DataFrame({
                "material_id": [df.iloc[i]["material_id"] for i in idx],
                "pretty_formula": [df.iloc[i]["pretty_formula"] for i in idx],
                "elements": [json.dumps(df.iloc[i]["elements"]) for i in idx],
                "cif": [df.iloc[i]["cif_raw"] for i in idx],
                "spacegroup.number": [df.iloc[i]["spg_raw"] for i in idx],
                "spacegroup.number.conv": [df.iloc[i]["spg_conv"] for i in idx],
                "cif.conv": [df.iloc[i]["cif_conv"] for i in idx],
                self.target_key: [df.iloc[i][self.target_key] for i in idx],
            })

        _mk(id_train).to_csv(self.out_dir / "train.csv", index=False)
        _mk(id_val).to_csv(self.out_dir / "val.csv", index=False)
        _mk(id_test).to_csv(self.out_dir / "test.csv", index=False)
        print(f"✓ FlowMM → train/val/test CSVs (test hash={hash10([df.iloc[i]['material_id'] for i in id_test])})")

################################################################################
# CLI handling
################################################################################

FACTORIES = {
    "atomgpt": AtomGPTFactory,
    "cdvae": CDVAEFactory,
    "flowmm": FlowMMFactory,
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv-files", nargs="+", required=True)
    common.add_argument("--id-key", default="mat_id")
    common.add_argument("--target", dest="target_key", default="Tc")
    common.add_argument("--output", required=True)
    common.add_argument("--max-size", type=int, default=None)
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--val-ratio", type=float, default=0.1)
    common.add_argument("--test-ratio", type=float, default=0.1)

    p = argparse.ArgumentParser(
        prog="alexandria_preprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Pre‑process Alexandria CSVs into model‑specific formats while sharing a single deterministic train/val/test split.""",
        ),
    )
    sub = p.add_subparsers(dest="model", required=True)
    for m in FACTORIES:
        sub.add_parser(m, parents=[common], help=f"prepare data for {m}")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)

    csv_files = [Path(p).expanduser().resolve() for p in args.csv_files]
    print(f"CSV files: {', '.join(str(p) for p in csv_files)}")

    # 1) Collect dataset
    df = collect_records_from_csv(
        csv_files,
        id_key=args.id_key,
        target_key=args.target_key,
        max_size=args.max_size,
    )
    n = len(df)
    print(f"✓ Collected {n} usable rows")

    # 2) Split
    id_train, id_val, id_test = deterministic_split(n, args.val_ratio, args.test_ratio, args.seed)
    print(f"Split sizes  train:{len(id_train)}  val:{len(id_val)}  test:{len(id_test)}")

    # 3) Dispatch
    out_dir = Path(args.output).expanduser().resolve()
    FACTORIES[args.model](out_dir, args.target_key).dump(df, id_train, id_val, id_test)


if __name__ == "__main__":
    main()

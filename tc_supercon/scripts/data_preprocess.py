#!/usr/bin/env python3
"""
Unified dataset preprocessing for AtomGPT, CDVAE, and FlowMM.

Usage
-----
Choose a sub‑command:

    python data_preprocess.py atomgpt  --dataset dft_3d --output ./atomgpt_out  \
                                       --target Tc_supercon --max-size 1000 --seed 123

    python data_preprocess.py cdvae    --dataset dft_3d --output ./cdvae_out    \
                                       --target Tc_supercon --max-size 1000 --seed 123

    python data_preprocess.py flowmm   --dataset dft_3d --output ./flowmm_out   \
                                       --target Tc_supercon --max-size 1000 --seed 123

The same *shuffle‑based* split (controlled by --seed) is used for every
sub‑command, so each model sees identical train/val/test partitions.  POSCAR
files are always written.  CSV formats match the original reference scripts.

Abstract‑factory layout:

    ┌────────────────────────┐
    │   DataPrepFactory (ABC)│
    ├────────────────────────┤
    │ +write_outputs(df, …)  │  <- implemented differently per model
    └──────────▲─────────────┘
               │
     ┌─────────┼─────────┬─────────┐
     │         │         │         │
 AtomGPTFactory│ CDVAEFactory│ FlowMMFactory
               │         │         │
               ▼         ▼         ▼
      id_prop.csv   train/val/test train/val/test
                    (minimal cols) (extended cols)

"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from abc import ABC, abstractmethod
from hashlib import sha256
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
from tqdm import tqdm

from jarvis.db.figshare import data as jarvis_data
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

################################################################################
# Utility helpers
################################################################################

def deterministic_split(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Return (id_train, id_val, id_test) as lists of indices."""
    indices = list(range(n_samples))
    random.seed(seed)
    random.shuffle(indices)

    n_val = int(val_ratio * n_samples)
    n_test = int(test_ratio * n_samples)
    n_train = n_samples - n_val - n_test

    id_train = indices[:n_train]
    id_val = indices[n_train : n_train + n_val]
    id_test = indices[n_train + n_val :]
    return id_train, id_val, id_test


def hash10(values: List[str]) -> str:
    h = sha256()
    for v in values:
        h.update(v.encode())
        h.update(b",")
    return h.hexdigest()[:10]


def canonicalise(pmg_struct: Structure, symprec: float = 0.1):
    """Return (cif_raw, cif_conv, spg_num, spg_conv).  Never raises."""
    try:
        sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec)
        spg_num = sga.get_space_group_number()
        cif_conv = sga.get_conventional_standard_structure().to(fmt="cif")
        spg_conv = SpacegroupAnalyzer(
            Structure.from_str(cif_conv, fmt="cif"), symprec=symprec
        ).get_space_group_number()
        return pmg_struct.to(fmt="cif"), cif_conv, spg_num, spg_conv
    except Exception:
        return "", "", -1, -1

################################################################################
# Abstract‑factory base class
################################################################################

class DataPrepFactory(ABC):
    """Abstract product: writes model‑specific outputs."""

    def __init__(self, out_dir: Path, target_key: str):
        self.out_dir = out_dir
        self.target_key = target_key
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # Shared step – always write POSCARs
    def write_poscar_files(self, df: pd.DataFrame, id_all: List[int]) -> List[str]:
        """Write POSCARs for every index in *id_all*.  Returns relative paths."""
        rel_paths: List[str] = []
        for idx in id_all:
            row = df.iloc[idx]
            jid = row["material_id"]
            atoms = row["atoms_j"]  # jarvis.core.Atoms instance
            fname = f"{jid}.vasp"
            fpath = self.out_dir / fname
            Poscar(atoms).write_file(fpath)
            rel_paths.append(fname)
        return rel_paths

    @abstractmethod
    def write_outputs(
        self,
        df: pd.DataFrame,
        id_train: List[int],
        id_val: List[int],
        id_test: List[int],
    ) -> None:  # pragma: no cover
        ...

################################################################################
# Concrete factories
################################################################################

class AtomGPTFactory(DataPrepFactory):
    """Produces id_prop.csv compatible with AtomGPT."""

    def write_outputs(self, df, id_train, id_val, id_test):
        # POSCARs – write once for all ids
        id_all = id_test + id_train + id_val  # test FIRST as requested
        rel_paths = self.write_poscar_files(df, id_all)

        # Build id_prop.csv – *test* rows first so AtomGPT can treat them as test
        targets = [df.iloc[i][self.target_key] for i in id_all]
        id_prop = pd.DataFrame({"structure_path": rel_paths, self.target_key: targets})
        id_prop.to_csv(self.out_dir / "id_prop.csv", index=False, header=False)

        print(
            f"✓ AtomGPT: wrote {len(id_prop)} rows -> {self.out_dir / 'id_prop.csv'}"
        )
        print(f"   hash10(ids)={hash10(rel_paths)}")


class CDVAEFactory(DataPrepFactory):
    """Produces train/val/test CSVs (minimal columns) for CDVAE."""

    def write_outputs(self, df, id_train, id_val, id_test):
        # POSCARs – optional for CDVAE but user requested; reuse helper
        self.write_poscar_files(df, id_train + id_val + id_test)

        def make_df(id_lst: List[int]):
            return pd.DataFrame(
                {
                    "material_id": [df.iloc[i]["material_id"] for i in id_lst],
                    "cif": [df.iloc[i]["cif_raw"] for i in id_lst],
                    self.target_key: [df.iloc[i][self.target_key] for i in id_lst],
                }
            )

        make_df(id_train).to_csv(self.out_dir / "train.csv", index=False)
        make_df(id_val).to_csv(self.out_dir / "val.csv", index=False)
        make_df(id_test).to_csv(self.out_dir / "test.csv", index=False)

        print(
            "✓ CDVAE: wrote train/val/test CSVs in", self.out_dir,
            " hashes", hash10([df.iloc[i]["material_id"] for i in id_test])
        )


class FlowMMFactory(DataPrepFactory):
    """Produces train/val/test CSVs with extra descriptors for FlowMM."""

    def write_outputs(self, df, id_train, id_val, id_test):
        # POSCARs – again, for completeness
        self.write_poscar_files(df, id_train + id_val + id_test)

        def make_df(id_lst: List[int]):
            return pd.DataFrame(
                {
                    "material_id": [df.iloc[i]["material_id"] for i in id_lst],
                    "pretty_formula": [df.iloc[i]["pretty_formula"] for i in id_lst],
                    "elements": [json.dumps(df.iloc[i]["elements"]) for i in id_lst],
                    "cif": [df.iloc[i]["cif_raw"] for i in id_lst],
                    "spacegroup.number": [df.iloc[i]["spg_raw"] for i in id_lst],
                    "spacegroup.number.conv": [df.iloc[i]["spg_conv"] for i in id_lst],
                    "cif.conv": [df.iloc[i]["cif_conv"] for i in id_lst],
                    self.target_key: [df.iloc[i][self.target_key] for i in id_lst],
                }
            )

        make_df(id_train).to_csv(self.out_dir / "train.csv", index=False)
        make_df(id_val).to_csv(self.out_dir / "val.csv", index=False)
        make_df(id_test).to_csv(self.out_dir / "test.csv", index=False)

        print(
            "✓ FlowMM: wrote train/val/test CSVs in", self.out_dir,
            " hashes", hash10([df.iloc[i]["material_id"] for i in id_test])
        )

################################################################################
# Dataset acquisition (shared for all factories)
################################################################################


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

################################################################################
# CLI parsing and wiring
################################################################################

FACTORY_REGISTRY = {
    "atomgpt": AtomGPTFactory,
    "cdvae": CDVAEFactory,
    "flowmm": FlowMMFactory,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_preprocess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Create consistent train/val/test splits for AtomGPT, CDVAE, and FlowMM.
            The split is controlled by --seed; use the same seed to guarantee
            identical partitions across models.
            """
        ),
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True, help="JARVIS dataset name")
    common.add_argument("--target", dest="target_key", default="Tc_supercon")
    common.add_argument("--id-key", default="jid")
    common.add_argument("--output", required=True, help="Output directory")
    common.add_argument("--max-size", type=int, default=None)
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--val-ratio", type=float, default=0.1)
    common.add_argument("--test-ratio", type=float, default=0.1)

    sub = p.add_subparsers(dest="model", required=True)

    for name in FACTORY_REGISTRY:
        sub.add_parser(name, parents=[common], help=f"prepare data for {name}")

    return p


def main(argv: Optional[List[str]] = None) -> None:  # entry‑point
    args = build_parser().parse_args(argv)

    # 1) Collect data (shared)
    df = collect_records(
        dataset_name=args.dataset,
        id_key=args.id_key,
        target_key=args.target_key,
        max_size=args.max_size,
    )
    n_samples = len(df)
    print(f"✓ Collected {n_samples} usable entries")

    # 2) Compute deterministic split
    id_train, id_val, id_test = deterministic_split(
        n_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"Split sizes  train:{len(id_train)}  val:{len(id_val)}  test:{len(id_test)}"
    )

    # 3) Dispatch to the requested factory
    factory_cls = FACTORY_REGISTRY[args.model]
    factory = factory_cls(Path(args.output), args.target_key)
    factory.write_outputs(df, id_train, id_val, id_test)


if __name__ == "__main__":
    main()

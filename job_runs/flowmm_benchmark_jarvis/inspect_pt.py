#!/usr/bin/env python
"""
write_flowmm_benchmark.py

Create a JARVIS-style benchmark CSV with columns
    id,target,prediction
where `target` and `prediction` are POSCAR strings (newline-escaped).

The ground-truth and predicted structures are read with
flowmm.old_eval.core.get_Crystal_obj_lists(…).

Usage
-----
python write_flowmm_benchmark.py \
    --pt_path  PATH/TO/consolidated_reconstruct.pt \
    --output_csv  flowmm_submission.csv \
    [--multi_eval] \
    [--ground_truth_pt PATH/TO/gt.pt] \
    [--dump_json  all_splits.json]

If the Crystal objects do **not** contain a usable identifier, the script
falls back to sequential sample_0001, sample_0002, … IDs.
"""
from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms, pmg_to_atoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from flowmm.old_eval.core import get_Crystal_obj_lists


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def crystal_to_atoms(crys) -> Atoms:
    """FlowMM Crystal ➜ JARVIS Atoms."""
    return pmg_to_atoms(crys.structure)


def build_id(crys, fallback: str) -> str:
    """Return a string ID for this Crystal (path stem, name, …)."""
    for attr in ("id", "name", "path"):
        if hasattr(crys, attr):
            val = getattr(crys, attr)
            if isinstance(val, (str, Path)) and val:
                return Path(val).stem
    return fallback


def poscar_string(atoms: Atoms) -> str:
    """POSCAR string with newlines escaped (like the CDVAE script)."""
    return Poscar(atoms).to_string().replace("\n", r"\n")


def rms_check(target_atoms: list[Atoms], pred_atoms: list[Atoms]) -> float | None:
    """Anonymous RMS distance (debug sanity-check)."""
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rms_vals = []
    for tgt, pred in tqdm(zip(target_atoms, pred_atoms),
                          total=len(target_atoms),
                          desc="RMS-check"):
        try:
            score, _ = matcher.get_rms_anonymous(
                pred.pymatgen_converter(), tgt.pymatgen_converter()
            )
            if score is not None:
                rms_vals.append(score)
        except Exception:
            # ignore mismatches; just keep going
            pass
    if rms_vals:
        return float(np.mean(rms_vals))
    return None


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", required=True,
                        help="FlowMM consolidated_reconstruct.pt (or similar)")
    parser.add_argument("--output_csv", required=True,
                        help="Benchmark CSV to write")
    parser.add_argument("--multi_eval", action="store_true",
                        help="Set if the .pt was generated with --multi_eval")
    parser.add_argument("--ground_truth_pt",
                        help="Optional separate GT .pt "
                             "(only needed if you passed "
                             "--ground_truth_path to FlowMM)")
    parser.add_argument("--dump_json",
                        help="Optional JSON dump of {id: POSCAR} for debugging")
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    # Load FlowMM prediction / target structures                            #
    # --------------------------------------------------------------------- #
    print("CWD at start:", os.getcwd())
    pred_crys, gt_crys, _ = get_Crystal_obj_lists(
        Path(args.pt_path),
        args.multi_eval,
        Path(args.ground_truth_pt) if args.ground_truth_pt else None,
    )
    print("CWD before write:", os.getcwd())

    # Convert to JARVIS Atoms and build IDs
    pred_atoms = [crystal_to_atoms(c) for c in pred_crys]
    tgt_atoms  = [crystal_to_atoms(c) for c in gt_crys]

    ids = [build_id(c, f"sample_{i:04d}") for i, c in enumerate(gt_crys)]

    if not (len(pred_atoms) == len(tgt_atoms) == len(ids)):
        raise RuntimeError("Prediction / target length mismatch")

    # --------------------------------------------------------------------- #
    # Write CSV                                                             #
    # --------------------------------------------------------------------- #
    with open(args.output_csv, "w") as fh:
        fh.write("id,target,prediction\n")
        for jid, tgt, pred in zip(ids, tgt_atoms, pred_atoms):
            fh.write(f"{jid},{poscar_string(tgt)},{poscar_string(pred)}\n")
    print(f"[✓] CSV written to {args.output_csv} "
          f"({len(pred_atoms)} rows)")

    # Optional JSON dump (handy for diff-ing)
    if args.dump_json:
        dump = {
            jid: {
                "target": poscar_string(tgt),
                "prediction": poscar_string(pred),
            }
            for jid, tgt, pred in zip(ids, tgt_atoms, pred_atoms)
        }
        with open(args.dump_json, "w") as jf:
            json.dump(dump, jf)
        print(f"[✓] JSON dump saved to {args.dump_json}")

    # --------------------------------------------------------------------- #
    # Quick RMS sanity-check                                                #
    # --------------------------------------------------------------------- #
    print("\nRunning anonymous RMS sanity-check …")
    rms = rms_check(tgt_atoms, pred_atoms)
    if rms is not None:
        print(f"[✓] Mean anonymous RMS: {rms:.4f} Å over "
              f"{len(tgt_atoms)} structures")
    else:
        print("[!] RMS could not be computed for any structure")

if __name__ == "__main__":
    main()

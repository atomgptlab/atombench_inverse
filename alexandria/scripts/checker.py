#!/usr/bin/env python3
"""
verify that AtomGPT, CDVAE and FlowMM test‑set IDs are identical.

Example
-------
python check_splits.py \
    --atomgpt-dir ./atomgpt_out \
    --cdvae-dir   ./cdvae_out   \
    --flowmm-dir  ./flowmm_out  \
    --strict-order

If all three share exactly the same IDs (and, with --strict-order, the same
ordering), the script prints "✓ Splits are identical" and exits 0.  Otherwise
it prints a detailed diff and exits with status 1.

Dependencies
------------
    pip install pandas tabulate

Only *pandas* is strictly required, but *tabulate* makes the diff table
prettier; it falls back to plain text if not installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

try:
    from tabulate import tabulate

    def _pretty_table(rows, headers):
        return tabulate(rows, headers=headers, tablefmt="github")
except ImportError:  # pragma: no cover
    def _pretty_table(rows, headers):
        header_line = " | ".join(headers)
        lines = [header_line, "-" * len(header_line)]
        for row in rows:
            lines.append(" | ".join(map(str, row)))
        return "\n".join(lines)


################################################################################
# helpers
################################################################################

def test_ids_cdvae(dir_: Path) -> List[str]:
    return pd.read_csv(dir_ / "test.csv")[["material_id"]].squeeze().tolist()


def test_ids_flowmm(dir_: Path) -> List[str]:
    return pd.read_csv(dir_ / "test.csv")[["material_id"]].squeeze().tolist()


def test_ids_atomgpt(dir_: Path, n_test: int) -> List[str]:
    id_prop = pd.read_csv(dir_ / "id_prop.csv", header=None, names=["path", "target"])
    ids_with_ext = id_prop.path.head(n_test).tolist()
    return [Path(p).stem for p in ids_with_ext]  # strip .vasp extension


################################################################################
# main logic
################################################################################

def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify identical test splits across models")
    ap.add_argument("--atomgpt-dir", required=True, type=Path)
    ap.add_argument("--cdvae-dir", required=True, type=Path)
    ap.add_argument("--flowmm-dir", required=True, type=Path)
    ap.add_argument("--strict-order", action="store_true", help="Require identical ordering, not just identical sets")
    args = ap.parse_args(argv)

    ids_cdvae = test_ids_cdvae(args.cdvae_dir)
    ids_flowmm = test_ids_flowmm(args.flowmm_dir)

    if len(ids_cdvae) != len(ids_flowmm):
        print("Mismatch in test‑set lengths between CDVAE and FlowMM.", file=sys.stderr)
        sys.exit(1)

    ids_atomgpt = test_ids_atomgpt(args.atomgpt_dir, n_test=len(ids_cdvae))

    # Optionally enforce order
    if args.strict_order:
        equal = ids_cdvae == ids_flowmm == ids_atomgpt
    else:
        equal = set(ids_cdvae) == set(ids_flowmm) == set(ids_atomgpt)

    if equal:
        print("\u2713 Splits are identical ✅")
        sys.exit(0)

    # Otherwise, produce diff
    def diff(a, b):
        return sorted(list(set(a) ^ set(b)))

    rows = [
        ("CDVAE vs FlowMM", len(diff(ids_cdvae, ids_flowmm))),
        ("CDVAE vs AtomGPT", len(diff(ids_cdvae, ids_atomgpt))),
        ("FlowMM vs AtomGPT", len(diff(ids_flowmm, ids_atomgpt))),
    ]
    print("\u274C Splits are NOT identical\n")
    print(_pretty_table(rows, headers=["Pair", "# differing IDs"]))
    sys.exit(1)


if __name__ == "__main__":
    main()

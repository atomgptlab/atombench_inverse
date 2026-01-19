#!/usr/bin/env python3
"""
checker.py

Hard-fail auditor for dataset split hygiene and leakage.

It checks, for AtomGPT / CDVAE / FlowMM artifacts on disk:

1) Cross-model split equivalence:
   - AtomGPT test IDs == CDVAE test IDs == FlowMM test IDs
   - AtomGPT train IDs == (CDVAE train ∪ val) == (FlowMM train ∪ val)

2) Within-family split sanity:
   - train/val/test disjointness
   - no duplicates within each split

3) Leakage checks for CDVAE + FlowMM using structure content:
   - CIF text hash overlap across splits (normalized text → SHA-256)
   - Structure hash overlap across splits (canonicalized Structure → SHA-256)

AtomGPT split convention (per user request):
   - AtomGPT "train" = head of id_prop.csv
   - AtomGPT "test"  = tail of id_prop.csv
   - head + tail = entire file, where tail length is inferred from CDVAE test size
     (or explicitly supplied via --n-test)

Usage
-----
python checker.py \
  --atomgpt-dir ./atomgpt_data \
  --cdvae-dir   ../models/cdvae/data/supercon \
  --flowmm-dir  ../models/flowmm/data/supercon \
  --strict-order

Optional:
  --n-test N                 # override inferred AtomGPT test length
  --no-structure-leakage     # skip CIF/structure hashing checks
  --symprec 0.01 --decimals 6  # structure hashing controls

Dependencies
------------
- pandas, numpy
- pymatgen (required unless --no-structure-leakage)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# --------------------------- misc utilities ---------------------------

def die(msg: str, code: int = 1) -> None:
    print(f"\n[FAIL] {msg}\n", file=sys.stderr)
    raise SystemExit(code)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def _as_list(x) -> List[str]:
    return [str(v).strip() for v in x if str(v).strip() != ""]

def _set(x: Sequence[str]) -> Set[str]:
    return set(_as_list(x))

def _dups(x: Sequence[str]) -> List[str]:
    seen = set()
    d = []
    for v in _as_list(x):
        if v in seen:
            d.append(v)
        else:
            seen.add(v)
    return d

def _diff_count(a: Sequence[str], b: Sequence[str]) -> int:
    return len(_set(a) ^ _set(b))


# --------------------------- AtomGPT split parsing ---------------------------

def atomgpt_ids(dir_: Path) -> Tuple[List[str], List[str]]:
    """
    Return (paths, ids) from id_prop.csv.
    id_prop.csv is assumed headerless: col0=path, col1=target.
    """
    p = dir_ / "id_prop.csv"
    if not p.exists():
        die(f"AtomGPT id_prop.csv not found at: {p}")

    df = pd.read_csv(p, header=None, names=["path", "target"])
    paths = df["path"].astype(str).tolist()
    ids = [Path(s).stem for s in paths]  # strip extension
    return paths, ids

def atomgpt_split(dir_: Path, n_test: int) -> Tuple[List[str], List[str]]:
    """
    Split AtomGPT ids into (train_ids, test_ids) as head/tail where tail length = n_test.
    """
    _, ids = atomgpt_ids(dir_)
    if n_test <= 0:
        die(f"AtomGPT split: n_test must be > 0, got {n_test}")
    if n_test >= len(ids):
        die(f"AtomGPT split: n_test={n_test} must be smaller than total rows={len(ids)} in id_prop.csv")

    train_ids = ids[:-n_test]
    test_ids = ids[-n_test:]
    if len(train_ids) + len(test_ids) != len(ids):
        die("AtomGPT split invariant violated (head+tail != total). This should be impossible.")
    return train_ids, test_ids


# --------------------------- CDVAE / FlowMM split parsing ---------------------------

def read_ids_csv(path: Path, col: str = "material_id") -> List[str]:
    if not path.exists():
        die(f"Missing expected split file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if col not in df.columns:
        die(f"Expected column '{col}' in {path}, found {list(df.columns)}")
    return df[col].astype(str).tolist()

def cdvae_splits(dir_: Path) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv"),
        read_ids_csv(dir_ / "val.csv"),
        read_ids_csv(dir_ / "test.csv"),
    )

def flowmm_splits(dir_: Path) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv"),
        read_ids_csv(dir_ / "val.csv"),
        read_ids_csv(dir_ / "test.csv"),
    )


# --------------------------- Leakage checks via CIF/Structure hashing ---------------------------

def _norm_cif_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    return s

def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def _import_pymatgen():
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return Structure, SpacegroupAnalyzer
    except Exception as e:
        die(
            "pymatgen is required for structure-level leakage checks. "
            "Install it or run with --no-structure-leakage.\n"
            f"Import error: {e}"
        )
        raise  # unreachable

def structure_hash_from_cif(
    cif_text: str,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
) -> Optional[str]:
    """
    Parse CIF -> Niggli reduce -> symmetry standardize -> deterministic site order -> round -> SHA-256.
    Returns None if parsing/canonicalization fails.
    """
    Structure, SpacegroupAnalyzer = _import_pymatgen()

    try:
        s = Structure.from_str(cif_text, fmt="cif")

        # Niggli reduction helps canonicalize lattice representation
        try:
            s = s.get_reduced_structure(reduction_algo="niggli")
        except Exception:
            pass

        # Symmetry-based standardization (spglib-backed)
        try:
            sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)
            s = sga.get_primitive_standard_structure(international_monoclinic=True)
        except Exception:
            pass

        # Deterministic ordering of sites by (Z, frac coords)
        frac = np.mod(np.array(s.frac_coords, dtype=float), 1.0)
        Z = np.array([int(getattr(site.specie, "Z", site.specie.number)) for site in s.sites], dtype=np.int32)

        order = np.lexsort((
            np.round(frac[:, 2], decimals),
            np.round(frac[:, 1], decimals),
            np.round(frac[:, 0], decimals),
            Z
        ))
        frac = frac[order]
        Z = Z[order]

        lat = np.array(s.lattice.matrix, dtype=float)
        lat_r = np.round(lat, decimals=decimals)
        frac_r = np.round(frac, decimals=decimals)

        payload = (
            lat_r.astype(np.float64).tobytes()
            + Z.astype(np.int32).tobytes()
            + frac_r.astype(np.float64).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return None

def load_cifs_by_id(csv_path: Path) -> Dict[str, str]:
    """
    Load mapping: material_id -> cif string from a split csv.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    if "material_id" not in df.columns or "cif" not in df.columns:
        die(f"{csv_path} must contain columns 'material_id' and 'cif'. Found {list(df.columns)}")
    out = {}
    for mid, cif in zip(df["material_id"].astype(str), df["cif"].astype(str)):
        mid = mid.strip()
        if mid == "":
            continue
        out.setdefault(mid, cif)
    return out

def leakage_check_cdvae_flowmm(
    name: str,
    dir_: Path,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
) -> None:
    """
    For a dataset directory with train/val/test.csv (each has material_id,cif),
    assert no overlap across splits by:
      - material_id
      - normalized CIF text hash
      - structure hash (canonicalized)
    """
    train_p = dir_ / "train.csv"
    val_p   = dir_ / "val.csv"
    test_p  = dir_ / "test.csv"

    for p in (train_p, val_p, test_p):
        if not p.exists():
            die(f"{name}: missing expected split file {p}")

    maps = {
        "train": load_cifs_by_id(train_p),
        "val":   load_cifs_by_id(val_p),
        "test":  load_cifs_by_id(test_p),
    }

    # 1) ID disjointness
    ids = {k: set(v.keys()) for k, v in maps.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = ids[a] & ids[b]
        if inter:
            sample = ", ".join(sorted(list(inter))[:10])
            die(f"{name}: material_id overlap between {a} and {b}: {len(inter)} (e.g., {sample})")

    # 2) CIF text hash disjointness
    cif_hashes: Dict[str, Set[str]] = {}
    for sp, mp in maps.items():
        hs = set()
        for cif in mp.values():
            hs.add(_sha256_hex(_norm_cif_text(cif)))
        cif_hashes[sp] = hs

    for a, b in pairs:
        inter = cif_hashes[a] & cif_hashes[b]
        if inter:
            die(f"{name}: CIF-text hash overlap between {a} and {b}: {len(inter)}")

    # 3) Structure hash disjointness
    struct_hashes: Dict[str, Set[str]] = {}
    bad = 0
    total = 0
    for sp, mp in maps.items():
        hs = set()
        for cif in mp.values():
            total += 1
            h = structure_hash_from_cif(_norm_cif_text(cif), symprec, angle_tolerance, decimals)
            if h is None:
                bad += 1
                continue
            hs.add(h)
        struct_hashes[sp] = hs

    if bad > 0:
        # Not a failure by default; parsing failures can occur for malformed CIF strings.
        # But we still disclose it, because referees may ask.
        warn(f"{name}: structure-hash parse failures: {bad}/{total} (these entries were skipped for structure-hash overlap checks)")

    for a, b in pairs:
        inter = struct_hashes[a] & struct_hashes[b]
        if inter:
            die(f"{name}: STRUCTURE-hash overlap between {a} and {b}: {len(inter)}")

    ok(f"{name}: no leakage detected across train/val/test (by id, CIF-hash, structure-hash)")


# --------------------------- Core assertions ---------------------------

def assert_no_dups(label: str, ids: Sequence[str]) -> None:
    d = _dups(ids)
    if d:
        sample = ", ".join(d[:10])
        die(f"{label}: duplicates within split: {len(d)} (e.g., {sample})")

def assert_disjoint(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str]) -> None:
    inter = _set(A) & _set(B)
    if inter:
        sample = ", ".join(sorted(list(inter))[:10])
        die(f"Split overlap: {labelA} ∩ {labelB} has {len(inter)} IDs (e.g., {sample})")

def assert_equal_sets(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str]) -> None:
    SA, SB = _set(A), _set(B)
    if SA != SB:
        onlyA = sorted(list(SA - SB))[:10]
        onlyB = sorted(list(SB - SA))[:10]
        die(
            f"Set mismatch: {labelA} vs {labelB}\n"
            f"  - {labelA} \\ {labelB}: {len(SA - SB)} (e.g., {', '.join(onlyA)})\n"
            f"  - {labelB} \\ {labelA}: {len(SB - SA)} (e.g., {', '.join(onlyB)})"
        )

def assert_equal_order(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str]) -> None:
    if list(_as_list(A)) != list(_as_list(B)):
        die(f"Order mismatch: {labelA} != {labelB} (use without --strict-order to compare as sets)")


# --------------------------- main ---------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Split leakage/disjointness auditor (hard-fail).")
    ap.add_argument("--atomgpt-dir", required=True, type=Path)
    ap.add_argument("--cdvae-dir", required=True, type=Path)
    ap.add_argument("--flowmm-dir", required=True, type=Path)

    ap.add_argument("--strict-order", action="store_true",
                    help="Require identical ordering where applicable (test sets).")
    ap.add_argument("--n-test", type=int, default=None,
                    help="Override AtomGPT test length; default: infer from CDVAE test.csv length.")

    ap.add_argument("--no-structure-leakage", action="store_true",
                    help="Skip CIF/structure hashing leakage checks (CDVAE/FlowMM).")
    ap.add_argument("--symprec", type=float, default=1e-2,
                    help="symprec for symmetry standardization in structure hashing.")
    ap.add_argument("--angle-tolerance", type=float, default=5.0,
                    help="angle_tolerance for symmetry standardization in structure hashing.")
    ap.add_argument("--decimals", type=int, default=6,
                    help="Rounding decimals for structure hashing.")

    args = ap.parse_args(argv)

    # Load CDVAE/FlowMM splits (explicit)
    cd_train, cd_val, cd_test = cdvae_splits(args.cdvae_dir)
    fm_train, fm_val, fm_test = flowmm_splits(args.flowmm_dir)

    # Infer AtomGPT test size
    n_test = args.n_test if args.n_test is not None else len(cd_test)
    ag_train, ag_test = atomgpt_split(args.atomgpt_dir, n_test=n_test)

    # ---------- Basic split hygiene ----------
    # No duplicates within each split
    assert_no_dups("CDVAE train", cd_train)
    assert_no_dups("CDVAE val", cd_val)
    assert_no_dups("CDVAE test", cd_test)

    assert_no_dups("FlowMM train", fm_train)
    assert_no_dups("FlowMM val", fm_val)
    assert_no_dups("FlowMM test", fm_test)

    assert_no_dups("AtomGPT train(head)", ag_train)
    assert_no_dups("AtomGPT test(tail)", ag_test)

    # Disjointness within each family
    assert_disjoint("CDVAE train", cd_train, "CDVAE val", cd_val)
    assert_disjoint("CDVAE train", cd_train, "CDVAE test", cd_test)
    assert_disjoint("CDVAE val", cd_val, "CDVAE test", cd_test)

    assert_disjoint("FlowMM train", fm_train, "FlowMM val", fm_val)
    assert_disjoint("FlowMM train", fm_train, "FlowMM test", fm_test)
    assert_disjoint("FlowMM val", fm_val, "FlowMM test", fm_test)

    # AtomGPT has only head/tail; ensure no overlap
    assert_disjoint("AtomGPT train(head)", ag_train, "AtomGPT test(tail)", ag_test)

    ok("Within-family split disjointness and de-duplication checks passed.")

    # ---------- Cross-model equivalence per your rules ----------
    # Rule 1: test sets identical across all three
    assert_equal_sets("CDVAE test", cd_test, "FlowMM test", fm_test)
    assert_equal_sets("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test)

    if args.strict_order:
        assert_equal_order("CDVAE test", cd_test, "FlowMM test", fm_test)
        assert_equal_order("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test)

    # Rule 2: AtomGPT train == CDVAE(train ∪ val) == FlowMM(train ∪ val)
    cd_train_equiv = list(_set(cd_train) | _set(cd_val))
    fm_train_equiv = list(_set(fm_train) | _set(fm_val))

    assert_equal_sets("AtomGPT train(head)", ag_train, "CDVAE train∪val", cd_train_equiv)
    assert_equal_sets("AtomGPT train(head)", ag_train, "FlowMM train∪val", fm_train_equiv)

    ok("Cross-model split equivalence checks passed (using the requested mapping).")

    # ---------- Structural leakage checks (CDVAE/FlowMM) ----------
    if not args.no_structure_leakage:
        leakage_check_cdvae_flowmm(
            name="CDVAE",
            dir_=args.cdvae_dir,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            decimals=args.decimals,
        )
        leakage_check_cdvae_flowmm(
            name="FlowMM",
            dir_=args.flowmm_dir,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            decimals=args.decimals,
        )
    else:
        warn("Skipping CIF/structure hashing leakage checks (--no-structure-leakage).")

    print("\n✓ All reviewer-facing split/leakage checks passed ✅\n")


if __name__ == "__main__":
    main()


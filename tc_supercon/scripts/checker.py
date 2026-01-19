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

Key behavior:
- Do NOT error on the first detected duplicate/leak.
- Instead:
  * compute how many overlaps exist
  * print overlapping hash tokens AND example IDs per split where they appear
  * check ALL datasets (CDVAE then FlowMM) and ONLY THEN exit non-zero if any issues exist

AtomGPT split convention (per user request):
   - AtomGPT "train" = head of id_prop.csv
   - AtomGPT "test"  = tail of id_prop.csv
   - head + tail = entire file, where tail length is inferred from CDVAE test size
     (or explicitly supplied via --n-test)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Issues encountered while parsing CIF: .* rounded to ideal values .*",
    category=UserWarning,
)

# --------------------------- misc utilities ---------------------------

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def _as_list(x) -> List[str]:
    return [str(v).strip() for v in x if str(v).strip() != ""]

def _set(x: Sequence[str]) -> Set[str]:
    return set(_as_list(x))

def _dups(x: Sequence[str]) -> List[str]:
    """Return duplicates (by value) preserving encounter order."""
    seen = set()
    d = []
    for v in _as_list(x):
        if v in seen:
            d.append(v)
        else:
            seen.add(v)
    return d

def _overlap_pairs() -> List[Tuple[str, str]]:
    return [("train", "val"), ("train", "test"), ("val", "test")]

def _short_ids(ids: List[str], k: int = 3) -> str:
    ids = [i for i in ids if i]
    if not ids:
        return "[]"
    shown = ids[:k]
    extra = len(ids) - len(shown)
    if extra > 0:
        return "[" + ", ".join(shown) + f", ... +{extra}" + "]"
    return "[" + ", ".join(shown) + "]"


# --------------------------- Error accumulator ---------------------------

@dataclass
class IssueLog:
    hard_failures: List[str]
    leak_summaries: List[str]
    leak_hashes: List[str]

    def __init__(self):
        self.hard_failures = []
        self.leak_summaries = []
        self.leak_hashes = []

    def add_fail(self, msg: str) -> None:
        self.hard_failures.append(msg)

    def add_leak_summary(self, msg: str) -> None:
        self.leak_summaries.append(msg)

    def add_leak_hash_line(self, msg: str) -> None:
        self.leak_hashes.append(msg)

    def any_fail(self) -> bool:
        return bool(self.hard_failures or self.leak_summaries or self.leak_hashes)

    def report_and_exit(self) -> None:
        if not self.any_fail():
            print("\n✓ All reviewer-facing split/leakage checks passed ✅\n")
            raise SystemExit(0)

        print("\n==================== SPLIT/LEAKAGE AUDIT: FAIL ====================", file=sys.stderr)
        if self.hard_failures:
            print("\n[HARD FAILURES]", file=sys.stderr)
            for msg in self.hard_failures:
                print(f"- {msg}", file=sys.stderr)

        if self.leak_summaries or self.leak_hashes:
            print("\n[LEAKAGE SUMMARY]", file=sys.stderr)
            for msg in self.leak_summaries:
                print(f"- {msg}", file=sys.stderr)

            if self.leak_hashes:
                print("\n[OVERLAPPING HASH TOKENS]", file=sys.stderr)
                for line in self.leak_hashes:
                    print(line, file=sys.stderr)

        print("\n====================================================================\n", file=sys.stderr)
        raise SystemExit(1)


# --------------------------- AtomGPT split parsing ---------------------------

def atomgpt_ids(dir_: Path, issues: IssueLog) -> Tuple[List[str], List[str]]:
    """
    Return (paths, ids) from id_prop.csv.
    id_prop.csv is assumed headerless: col0=path, col1=target.
    """
    p = dir_ / "id_prop.csv"
    if not p.exists():
        issues.add_fail(f"AtomGPT id_prop.csv not found at: {p}")
        return [], []

    df = pd.read_csv(p, header=None, names=["path", "target"])
    paths = df["path"].astype(str).tolist()
    ids = [Path(s).stem for s in paths]  # strip extension
    return paths, ids

def atomgpt_split(dir_: Path, n_test: int, issues: IssueLog) -> Tuple[List[str], List[str]]:
    """
    Split AtomGPT ids into (train_ids, test_ids) as head/tail where tail length = n_test.
    """
    _, ids = atomgpt_ids(dir_, issues)
    if not ids:
        return [], []

    if n_test <= 0:
        issues.add_fail(f"AtomGPT split: n_test must be > 0, got {n_test}")
        return [], []
    if n_test >= len(ids):
        issues.add_fail(
            f"AtomGPT split: n_test={n_test} must be smaller than total rows={len(ids)} in id_prop.csv"
        )
        return [], []

    train_ids = ids[:-n_test]
    test_ids = ids[-n_test:]
    if len(train_ids) + len(test_ids) != len(ids):
        issues.add_fail("AtomGPT split invariant violated (head+tail != total).")
        return [], []

    return train_ids, test_ids


# --------------------------- CDVAE / FlowMM split parsing ---------------------------

def read_ids_csv(path: Path, issues: IssueLog, col: str = "material_id") -> List[str]:
    if not path.exists():
        issues.add_fail(f"Missing expected split file: {path}")
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if col not in df.columns:
        issues.add_fail(f"Expected column '{col}' in {path}, found {list(df.columns)}")
        return []
    return df[col].astype(str).tolist()

def cdvae_splits(dir_: Path, issues: IssueLog) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv", issues),
        read_ids_csv(dir_ / "val.csv", issues),
        read_ids_csv(dir_ / "test.csv", issues),
    )

def flowmm_splits(dir_: Path, issues: IssueLog) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv", issues),
        read_ids_csv(dir_ / "val.csv", issues),
        read_ids_csv(dir_ / "test.csv", issues),
    )


# --------------------------- Leakage checks via CIF/Structure hashing ---------------------------

def _norm_cif_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    return s

def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def _import_pymatgen(issues: IssueLog):
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return Structure, SpacegroupAnalyzer
    except Exception as e:
        issues.add_fail(
            "pymatgen is required for structure-level leakage checks. "
            "Install it or run with --no-structure-leakage.\n"
            f"Import error: {e}"
        )
        return None, None

def structure_hash_from_cif(
    cif_text: str,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    issues: IssueLog,
) -> Optional[str]:
    """
    Parse CIF -> Niggli reduce -> symmetry standardize -> deterministic site order -> round -> SHA-256.
    Returns None if parsing/canonicalization fails.
    """
    Structure, SpacegroupAnalyzer = _import_pymatgen(issues)
    if Structure is None:
        return None

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

def load_cifs_by_id(csv_path: Path, issues: IssueLog) -> Dict[str, str]:
    """
    Load mapping: material_id -> cif string from a split csv.
    """
    if not csv_path.exists():
        issues.add_fail(f"Missing expected split file: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    if "material_id" not in df.columns or "cif" not in df.columns:
        issues.add_fail(
            f"{csv_path} must contain columns 'material_id' and 'cif'. Found {list(df.columns)}"
        )
        return {}

    out: Dict[str, str] = {}
    for mid, cif in zip(df["material_id"].astype(str), df["cif"].astype(str)):
        mid = mid.strip()
        if mid == "":
            continue
        out.setdefault(mid, cif)
    return out

def _collect_overlapping_tokens(
    tokens_by_split: Dict[str, Set[str]]
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Return:
      - union_overlaps: all tokens that overlap across ANY split pair
      - pair_to_tokens: mapping "a|b" -> overlapping token set
    """
    pair_to_tokens: Dict[str, Set[str]] = {}
    union_overlaps: Set[str] = set()
    for a, b in _overlap_pairs():
        inter = tokens_by_split.get(a, set()) & tokens_by_split.get(b, set())
        key = f"{a}|{b}"
        if inter:
            pair_to_tokens[key] = inter
            union_overlaps |= inter
    return union_overlaps, pair_to_tokens

def _invert_index(token_to_ids: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # already token->ids; just defensive normalize
    out: Dict[str, List[str]] = {}
    for tok, ids in token_to_ids.items():
        out[tok] = [str(i) for i in ids]
    return out

def leakage_check_cdvae_flowmm(
    name: str,
    dir_: Path,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    issues: IssueLog,
) -> None:
    """
    For a dataset directory with train/val/test.csv (each has material_id,cif),
    check overlap across splits by:
      - material_id
      - normalized CIF text hash
      - structure hash (canonicalized)

    Output policy (per request):
      - print hash tokens AND example IDs per split (no snippets).
      - accumulate results; do not early-exit.
    """
    maps = {
        "train": load_cifs_by_id(dir_ / "train.csv", issues),
        "val":   load_cifs_by_id(dir_ / "val.csv", issues),
        "test":  load_cifs_by_id(dir_ / "test.csv", issues),
    }

    # If reading failed badly, don't cascade; issues already recorded.
    if not maps["train"] and not maps["val"] and not maps["test"]:
        return

    # 1) material_id overlap (hard leakage)
    ids = {k: set(v.keys()) for k, v in maps.items()}
    for a, b in _overlap_pairs():
        inter = ids.get(a, set()) & ids.get(b, set())
        if inter:
            issues.add_leak_summary(f"{name}: material_id overlap between {a} and {b}: {len(inter)}")

    # 2) CIF text hash overlap (strict identity of normalized text)
    def cif_hash(cif: str) -> str:
        return _sha256_hex(_norm_cif_text(cif))

    cif_tokens_by_split: Dict[str, Set[str]] = {}
    cif_tok_to_ids_by_split: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}

    for sp, mp in maps.items():
        toks: Set[str] = set()
        tok_to_ids: Dict[str, List[str]] = {}
        for mid, cif in mp.items():
            tok = cif_hash(cif)
            toks.add(tok)
            tok_to_ids.setdefault(tok, []).append(mid)
        cif_tokens_by_split[sp] = toks
        cif_tok_to_ids_by_split[sp] = _invert_index(tok_to_ids)

    cif_union, cif_pair_to = _collect_overlapping_tokens(cif_tokens_by_split)
    if cif_union:
        issues.add_leak_summary(f"{name}: CIF-text hash overlaps across splits: {len(cif_union)}")
        tok_to_pairs: Dict[str, List[str]] = {t: [] for t in cif_union}
        for pair, toks in cif_pair_to.items():
            for t in toks:
                tok_to_pairs.setdefault(t, []).append(pair)

        for t in sorted(tok_to_pairs.keys()):
            pairs = ",".join(sorted(tok_to_pairs[t]))
            train_ids = cif_tok_to_ids_by_split["train"].get(t, [])
            val_ids = cif_tok_to_ids_by_split["val"].get(t, [])
            test_ids = cif_tok_to_ids_by_split["test"].get(t, [])
            issues.add_leak_hash_line(
                f"{name} CIF  {t}  splits={pairs}  "
                f"train={_short_ids(train_ids)}  val={_short_ids(val_ids)}  test={_short_ids(test_ids)}"
            )

    # 3) STRUCTURE hash overlap (tolerant canonicalization; controlled by symprec/decimals)
    bad = 0
    total = 0
    struct_tokens_by_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    struct_tok_to_ids_by_split: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}

    for sp, mp in maps.items():
        hs: Set[str] = set()
        tok_to_ids: Dict[str, List[str]] = {}
        for mid, cif in mp.items():
            total += 1
            h = structure_hash_from_cif(_norm_cif_text(cif), symprec, angle_tolerance, decimals, issues)
            if h is None:
                bad += 1
                continue
            hs.add(h)
            tok_to_ids.setdefault(h, []).append(mid)
        struct_tokens_by_split[sp] = hs
        struct_tok_to_ids_by_split[sp] = _invert_index(tok_to_ids)

    if bad > 0:
        warn(f"{name}: structure-hash parse failures: {bad}/{total} (skipped for STRUCTURE hash overlap checks)")

    struct_union, struct_pair_to = _collect_overlapping_tokens(struct_tokens_by_split)
    if struct_union:
        issues.add_leak_summary(f"{name}: STRUCTURE-hash overlaps across splits: {len(struct_union)}")
        tok_to_pairs: Dict[str, List[str]] = {t: [] for t in struct_union}
        for pair, toks in struct_pair_to.items():
            for t in toks:
                tok_to_pairs.setdefault(t, []).append(pair)

        for t in sorted(tok_to_pairs.keys()):
            pairs = ",".join(sorted(tok_to_pairs[t]))
            train_ids = struct_tok_to_ids_by_split["train"].get(t, [])
            val_ids = struct_tok_to_ids_by_split["val"].get(t, [])
            test_ids = struct_tok_to_ids_by_split["test"].get(t, [])
            issues.add_leak_hash_line(
                f"{name} STRC {t}  splits={pairs}  "
                f"train={_short_ids(train_ids)}  val={_short_ids(val_ids)}  test={_short_ids(test_ids)}"
            )

    if not (cif_union or struct_union) and all(
        (ids.get(a, set()) & ids.get(b, set()) == set()) for a, b in _overlap_pairs()
    ):
        ok(f"{name}: no leakage detected across train/val/test (by id, CIF-hash, structure-hash)")


# --------------------------- Core assertions (accumulating) ---------------------------

def assert_no_dups(label: str, ids: Sequence[str], issues: IssueLog) -> None:
    d = _dups(ids)
    if d:
        uniq = sorted(list(set(d)))
        issues.add_fail(f"{label}: duplicates within split: {len(d)} (unique duplicated IDs: {len(uniq)})")

def assert_disjoint(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> None:
    inter = _set(A) & _set(B)
    if inter:
        issues.add_fail(f"Split overlap: {labelA} ∩ {labelB} has {len(inter)} IDs")

def assert_equal_sets(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> None:
    SA, SB = _set(A), _set(B)
    if SA != SB:
        issues.add_fail(
            f"Set mismatch: {labelA} vs {labelB}  "
            f"({len(SA - SB)} only-in-{labelA}, {len(SB - SA)} only-in-{labelB})"
        )

def assert_equal_order(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> None:
    if list(_as_list(A)) != list(_as_list(B)):
        issues.add_fail(f"Order mismatch: {labelA} != {labelB} (use without --strict-order to compare as sets)")


# --------------------------- main ---------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Split leakage/disjointness auditor (hard-fail after full run).")
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
    issues = IssueLog()

    # Load CDVAE/FlowMM splits (explicit)
    cd_train, cd_val, cd_test = cdvae_splits(args.cdvae_dir, issues)
    fm_train, fm_val, fm_test = flowmm_splits(args.flowmm_dir, issues)

    # Infer AtomGPT test size
    n_test = args.n_test if args.n_test is not None else len(cd_test)
    ag_train, ag_test = atomgpt_split(args.atomgpt_dir, n_test=n_test, issues=issues)

    # ---------- Basic split hygiene ----------
    assert_no_dups("CDVAE train", cd_train, issues)
    assert_no_dups("CDVAE val", cd_val, issues)
    assert_no_dups("CDVAE test", cd_test, issues)

    assert_no_dups("FlowMM train", fm_train, issues)
    assert_no_dups("FlowMM val", fm_val, issues)
    assert_no_dups("FlowMM test", fm_test, issues)

    assert_no_dups("AtomGPT train(head)", ag_train, issues)
    assert_no_dups("AtomGPT test(tail)", ag_test, issues)

    assert_disjoint("CDVAE train", cd_train, "CDVAE val", cd_val, issues)
    assert_disjoint("CDVAE train", cd_train, "CDVAE test", cd_test, issues)
    assert_disjoint("CDVAE val", cd_val, "CDVAE test", cd_test, issues)

    assert_disjoint("FlowMM train", fm_train, "FlowMM val", fm_val, issues)
    assert_disjoint("FlowMM train", fm_train, "FlowMM test", fm_test, issues)
    assert_disjoint("FlowMM val", fm_val, "FlowMM test", fm_test, issues)

    assert_disjoint("AtomGPT train(head)", ag_train, "AtomGPT test(tail)", ag_test, issues)

    if not issues.hard_failures:
        ok("Within-family split disjointness and de-duplication checks passed.")

    # ---------- Cross-model equivalence per requested rules ----------
    assert_equal_sets("CDVAE test", cd_test, "FlowMM test", fm_test, issues)
    assert_equal_sets("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test, issues)

    if args.strict_order:
        assert_equal_order("CDVAE test", cd_test, "FlowMM test", fm_test, issues)
        assert_equal_order("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test, issues)

    cd_train_equiv = list(_set(cd_train) | _set(cd_val))
    fm_train_equiv = list(_set(fm_train) | _set(fm_val))

    assert_equal_sets("AtomGPT train(head)", ag_train, "CDVAE train∪val", cd_train_equiv, issues)
    assert_equal_sets("AtomGPT train(head)", ag_train, "FlowMM train∪val", fm_train_equiv, issues)

    if not any("Set mismatch" in s or "Order mismatch" in s for s in issues.hard_failures):
        ok("Cross-model split equivalence checks passed (using the requested mapping).")

    # ---------- Structural leakage checks (CDVAE/FlowMM) ----------
    if not args.no_structure_leakage:
        leakage_check_cdvae_flowmm(
            name="CDVAE",
            dir_=args.cdvae_dir,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            decimals=args.decimals,
            issues=issues,
        )
        leakage_check_cdvae_flowmm(
            name="FlowMM",
            dir_=args.flowmm_dir,
            symprec=args.symprec,
            angle_tolerance=args.angle_tolerance,
            decimals=args.decimals,
            issues=issues,
        )
    else:
        warn("Skipping CIF/structure hashing leakage checks (--no-structure-leakage).")

    issues.report_and_exit()


if __name__ == "__main__":
    main()


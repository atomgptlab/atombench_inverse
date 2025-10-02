#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from jarvis.io.vasp.inputs import Poscar

# ---------- POSCAR parsing & canonicalization ----------

def _unescape_poscar_text(s: str) -> str:
    """Turn CSV-literal text into a parseable POSCAR string."""
    # normalize newlines; interpret literal backslash escapes
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n").replace("\\t", " ")
    return s.strip()

def _parse_poscar_atoms(poscar_text: str):
    """
    Parse POSCAR text -> jarvis.core.atoms.Atoms (or None on failure).
    """
    try:
        p = Poscar.from_string(_unescape_poscar_text(poscar_text))
        return p.atoms
    except Exception:
        return None

def _get_species_and_frac(atoms):
    """
    Extract (species_list, frac_coords ndarray) from a JARVIS Atoms object.
    We try to be defensive about attribute names.
    """
    # species
    species = getattr(atoms, "elements", None)
    if species is None:
        # fallback: composition or sites
        try:
            species = [s.specie for s in atoms]
        except Exception:
            raise ValueError("Could not read species from Atoms.")

    # fractional coordinates
    frac = getattr(atoms, "frac_coords", None)
    if frac is None:
        # Try to derive fractional from cartesian and lattice
        lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
        cart = np.asarray(getattr(atoms, "coords", None), dtype=float)
        if lat is None or cart is None:
            raise ValueError("Could not read coordinates from Atoms.")
        # Convert cartesian to fractional: frac = cart * inv(lat.T)
        # (jarvis uses row vectors for coords)
        inv_lat_T = np.linalg.inv(lat.T)
        frac = cart @ inv_lat_T

    return list(species), np.asarray(frac, dtype=float)

def _structure_signature(atoms, decimals: int = 1):
    """
    Build an order-insensitive signature of a structure with rounding.
    - Lattice matrix rounded to 'decimals'
    - Fractional coordinates rounded to 'decimals'
    - Species order ignored: coords grouped per species and each group's coords sorted
    Returns a hashable tuple.
    """
    if atoms is None:
        return ("__PARSE_FAILED__",)

    # Lattice
    lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
    if lat is None or lat.shape != (3, 3):
        return ("__BAD_LATTICE__",)
    lat_r = np.round(lat, decimals)

    # Species + frac coords
    species, frac = _get_species_and_frac(atoms)
    if len(species) != len(frac):
        return ("__LEN_MISMATCH__",)

    frac_r = np.round(frac, decimals)

    # Group by species -> multiset of coord triplets (sorted)
    buckets: Dict[str, List[Tuple[float, float, float]]] = {}
    for sp, xyz in zip(species, frac_r):
        tpl = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        buckets.setdefault(str(sp), []).append(tpl)

    # Sort each species bucket's coordinates for order-insensitive compare
    for sp in buckets:
        buckets[sp].sort()

    # Build signature: (rounded lattice, sorted list of (species, tuple(coords...)))
    lat_sig = tuple(lat_r.reshape(-1).tolist())
    species_sig = tuple(sorted((sp, tuple(coords)) for sp, coords in buckets.items()))
    return ("OK", lat_sig, species_sig)

# ---------- Original file-discovery & grouping logic ----------

def canonicalize_target_struct(s: Optional[str]) -> Tuple:
    """Convert TARGET string to an order-insensitive rounded structural signature."""
    if s is None or str(s).strip() == "":
        return ("__EMPTY__",)
    atoms = _parse_poscar_atoms(str(s))
    return _structure_signature(atoms, decimals=1)

def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Return the CSV in dir_path (or its subdirs) that has the expected header."""
    candidates: List[Tuple[float, Path]] = []
    for p, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".csv"):
                path = Path(p) / f
                try:
                    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                        reader = csv.DictReader(fh)
                        fields = [fn.strip().lower() for fn in (reader.fieldnames or [])]
                        if {"id", "target", "prediction"}.issubset(set(fields)):
                            candidates.append((path.stat().st_mtime, path))
                except Exception:
                    # Skip unreadable files
                    continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

def load_id_to_target(csv_path: Path) -> Dict[str, Tuple]:
    """Load mapping id -> structural signature (rounded, species-order-insensitive)."""
    mapping: Dict[str, Tuple] = {}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            _id = str(row["id"]).strip()
            tgt_sig = canonicalize_target_struct(row["target"])
            if _id in mapping and mapping[_id] != tgt_sig:
                raise ValueError(
                    f"Duplicate ID with differing TARGETs in {csv_path}: {_id}"
                )
            mapping[_id] = tgt_sig
    return mapping

def group_benchmarks(root: Path):
    """Group benchmark directories by dataset tag ('jarvis' or 'alex')."""
    groups = {"jarvis": [], "alex": []}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.lower()
        dataset = None
        if "jarvis" in name:
            dataset = "jarvis"
        elif "alexandria" in name or "alex" in name:
            dataset = "alex"
        if dataset is None:
            continue
        csv_path = find_benchmark_csv(entry)
        if csv_path is None:
            print(f"[WARN] No matching CSV in {entry}")
            continue
        groups[dataset].append((entry.name, csv_path))
    return groups

def compare_group(dataset_key: str, items: List[Tuple[str, Path]], show_diff: int) -> bool:
    """Compare ID sets and TARGET structures across CSVs for a dataset group."""
    pretty_name = "alexandria" if dataset_key == "alex" else "jarvis"
    if len(items) < 2:
        print(f"[INFO] Only {len(items)} benchmark found for {pretty_name}; cannot verify consensus.")
        return False

    # Load mappings
    loaded = []
    for label, path in items:
        try:
            mapping = load_id_to_target(path)
            loaded.append((label, path, mapping))
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return False

    # Compare ID sets
    id_sets = [set(m.keys()) for _, _, m in loaded]
    all_ids = id_sets[0]
    ids_match = all(s == all_ids for s in id_sets[1:])
    if not ids_match:
        print(f"[MISMATCH] ID sets differ for {pretty_name}.")
        union_ids = set().union(*id_sets)
        for label, path, mapping in loaded:
            missing = union_ids - set(mapping.keys())
            extra = set(mapping.keys()) - union_ids  # should be empty
            if missing:
                sample = ", ".join(sorted(list(missing))[:show_diff])
                print(f"  - {label}: missing {len(missing)} IDs (e.g., {sample})")
            if extra:
                sample = ", ".join(sorted(list(extra))[:show_diff])
                print(f"  - {label}: has {len(extra)} unexpected IDs (e.g., {sample})")
        return False

    # Compare TARGETs (using rounded, species-order-insensitive signatures)
    base_label, base_path, base_map = loaded[0]
    mismatches = []
    for _id in sorted(all_ids):
        base_sig = base_map[_id]
        for label, path, mapping in loaded[1:]:
            if mapping[_id] != base_sig:
                mismatches.append((_id, base_label, base_path, label, path))
                if len(mismatches) >= show_diff:
                    break
        if len(mismatches) >= show_diff:
            break

    if mismatches:
        print(f"[MISMATCH] TARGET structures differ for {pretty_name} "
              f"(rounded to 1 decimal, species-order-insensitive). Showing up to {show_diff}:")
        for _id, b_label, b_path, l_label, l_path in mismatches:
            print(f"  - ID {_id}: {b_label} ({b_path.name}) != {l_label} ({l_path.name})")
        return False

    print(f"IDs and TARGET structures match for {pretty_name} "
          f"(rounded to 1 decimal, species-order-insensitive).")
    return True

def main():
    ap = argparse.ArgumentParser(
        description=("Verify that benchmark CSVs share the same ID set and TARGET rows "
                     "(compared as structures rounded to 1 decimal and species-order-insensitive).")
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("job_runs"),
        help="Path to the job_runs directory (default: ./job_runs)",
    )
    ap.add_argument(
        "--show-diff",
        type=int,
        default=5,
        help="Max number of example differences to display (default: 5)",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root path does not exist or is not a directory: {root}")
        raise SystemExit(2)

    groups = group_benchmarks(root)
    any_fail = False
    for dataset in ("jarvis", "alex"):
        items = groups.get(dataset, [])
        items.sort(key=lambda t: t[0].lower())
        ok = compare_group(dataset, items, args.show_diff)
        any_fail = any_fail or not ok

    raise SystemExit(1 if any_fail else 0)

if __name__ == "__main__":
    main()

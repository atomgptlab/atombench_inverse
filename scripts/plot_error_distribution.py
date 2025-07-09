#!/usr/bin/env python
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

from sklearn.metrics import mean_absolute_error
from scipy.stats import wasserstein_distance, entropy
from jarvis.io.vasp.inputs import Poscar
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from jarvis.core.atoms import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({"font.size": 18})

def emd_distance(p, q, bins=None):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    if bins is None:
        bins = np.arange(len(p))
    return wasserstein_distance(bins, bins, u_weights=p, v_weights=q)

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    return entropy(p, q)

# ── Load predictions CSV and parse POSCAR strings ────────────────────────────
df = pd.read_csv("AI-AtomGen-prop-dft_3d-test-rmse.csv")
records = []
for _, row in df.iterrows():
    tgt = Poscar.from_string(row['target'].replace("\\n", "\n")).atoms.to_dict()
    pred = Poscar.from_string(row['prediction'].replace("\\n", "\n")).atoms.to_dict()
    records.append({'target': tgt, 'predicted': pred})

# ── Extract lattice params and angles ───────────────────────────────────────
x_a = [r["target"]["abc"][0] for r in records];  y_a = [r["predicted"]["abc"][0] for r in records]
x_b = [r["target"]["abc"][1] for r in records];  y_b = [r["predicted"]["abc"][1] for r in records]
x_c = [r["target"]["abc"][2] for r in records];  y_c = [r["predicted"]["abc"][2] for r in records]
x_alpha = [r["target"]["angles"][0] for r in records];  y_alpha = [r["predicted"]["angles"][0] for r in records]
x_beta  = [r["target"]["angles"][1] for r in records];  y_beta  = [r["predicted"]["angles"][1] for r in records]
x_gamma = [r["target"]["angles"][2] for r in records];  y_gamma = [r["predicted"]["angles"][2] for r in records]

# ── Set up figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
grid = GridSpec(2, 3, figure=fig)

# ── (a) histogram of a ─────────────────────────────────────────────────────
plt.subplot(grid[0, 0])
wxa = np.ones_like(x_a)/len(x_a)*100;  wya = np.ones_like(y_a)/len(y_a)*100
plt.hist(x_a, bins=np.arange(2,7,0.1), weights=wxa, alpha=0.6, label="target")
plt.hist(y_a, bins=np.arange(2,7,0.1), weights=wya, alpha=0.6, label="predicted")
plt.xlabel(r"a ($\AA$)");  plt.title("(a)");  plt.ylabel("Materials dist.");  plt.legend()

# ── (b) histogram of b ─────────────────────────────────────────────────────
plt.subplot(grid[0, 1])
wxb = np.ones_like(x_b)/len(x_b)*100;  wyb = np.ones_like(y_b)/len(y_b)*100
plt.hist(x_b, bins=np.arange(2,7,0.1), weights=wxb, alpha=0.6, label="target_b")
plt.hist(y_b, bins=np.arange(2,7,0.1), weights=wyb, alpha=0.6, label="predicted_b")
plt.xlabel(r"b ($\AA$)");  plt.title("(b)")

# ── (c) histogram of gamma ─────────────────────────────────────────────────
plt.subplot(grid[0, 2])
wxg = np.ones_like(x_gamma)/len(x_gamma)*100;  wyg = np.ones_like(y_gamma)/len(y_gamma)*100
plt.hist(x_gamma, bins=np.arange(30,150,10), weights=wxg, alpha=0.6, label="target_gamma")
plt.hist(y_gamma, bins=np.arange(30,150,10), weights=wyg, alpha=0.6, label="predicted_gamma")
plt.xlabel(r"$\gamma$ ($^\circ$)");  plt.title("(c)")

# ── Composition, space group, Bravais lattice, weight collections ───────────
comp_same = []; comp_total = []
spg_same = []; spg_total = []
x_spg = []; y_spg = []
x_lat = []; y_lat = []
x_Z = []; y_Z = []

for rec in records:
    a1 = Atoms.from_dict(rec["target"])
    a2 = Atoms.from_dict(rec["predicted"])
    comp1 = a1.composition.reduced_formula
    comp2 = a2.composition.reduced_formula
    x_Z.append(a1.composition.weight);  y_Z.append(a2.composition.weight)

    # space group
    sga1 = SpacegroupAnalyzer(a1.pymatgen_converter(), symprec=0.1)
    sga2 = SpacegroupAnalyzer(a2.pymatgen_converter(), symprec=0.1)
    spg1 = sga1.get_space_group_number();  spg2 = sga2.get_space_group_number()
    x_spg.append(spg1);  y_spg.append(spg2)
    if spg1 == spg2: spg_same.append(rec)
    spg_total.append(rec)

    # Bravais lattice
    lat1 = Spacegroup3D(a1).crystal_system
    lat2 = Spacegroup3D(a2).crystal_system
    x_lat.append(lat1);  y_lat.append(lat2)

    if comp1 == comp2:
        comp_same.append(rec)
    comp_total.append(rec)

# ── Convert Bravais-lattice strings → categorical counts ─────────────────────
lat_order = [
    "triclinic","monoclinic","orthorhombic",
    "tetragonal","trigonal","hexagonal","cubic"
]
lat_to_idx = {name:i for i,name in enumerate(lat_order)}
x_lat_idx = np.array([lat_to_idx[l] for l in x_lat], dtype=int)
y_lat_idx = np.array([lat_to_idx[l] for l in y_lat], dtype=int)
x_lat_counts = np.bincount(x_lat_idx, minlength=len(lat_order))
y_lat_counts = np.bincount(y_lat_idx, minlength=len(lat_order))

# ── Compute aggregate metrics ───────────────────────────────────────────────
avg_kld = np.mean([
    kl_divergence(x_a,y_a), kl_divergence(x_b,y_b), kl_divergence(x_c,y_c),
    kl_divergence(x_alpha,y_alpha), kl_divergence(x_beta,y_beta), kl_divergence(x_gamma,y_gamma)
])
avg_mae = np.mean([
    mean_absolute_error(x_a,y_a), mean_absolute_error(x_b,y_b),
    mean_absolute_error(x_c,y_c), mean_absolute_error(x_alpha,y_alpha),
    mean_absolute_error(x_beta,y_beta), mean_absolute_error(x_gamma,y_gamma)
])

# ── (d) Spacegroup histogram ────────────────────────────────────────────────
plt.subplot(grid[1, 0])
wsx = np.ones_like(x_spg)/len(x_spg)*100;  wsy = np.ones_like(y_spg)/len(y_spg)*100
plt.hist(x_spg, bins=np.arange(0,231,10), weights=wsx, alpha=0.6, label="target_spg")
plt.hist(y_spg, bins=np.arange(0,231,10), weights=wsy, alpha=0.6, label="predicted_spg")
plt.ylabel("Materials dist."); plt.xlabel("Spacegroup number"); plt.title("(d)")

# ── (e) Bravais lattice bar chart with numeric x-axis ────────────────────────
plt.subplot(grid[1, 1])
width = 0.35
pos = np.arange(len(lat_order))
plt.bar(pos - width/2, x_lat_counts, width, alpha=0.6, label="target_lat")
plt.bar(pos + width/2, y_lat_counts, width, alpha=0.6, label="predicted_lat")
plt.xticks(pos, pos)  # numeric codes 0,1,...
plt.xlabel("Bravais lattice (code)"); plt.ylabel("Materials count"); plt.title("(e)"); plt.legend()

# ── (f) Weight histogram ────────────────────────────────────────────────────
plt.subplot(grid[1, 2])
wxz = np.ones_like(x_Z)/len(x_Z)*100;  wyz = np.ones_like(y_Z)/len(y_Z)*100
plt.hist(x_Z, bins=np.arange(15,2000,100), weights=wxz, alpha=0.6, label="target_wt")
plt.hist(y_Z, bins=np.arange(15,2000,100), weights=wyz, alpha=0.6, label="predicted_wt")
plt.xlabel("Weight (AMU)"); plt.title("(f)"); plt.legend()

# ── Final layout, title, save ───────────────────────────────────────────────
plt.tight_layout()
bnchmk_name_dict = {
    "agpt_benchmark_alex": "AtomGPT Alexandria",
    "agpt_benchmark_jarvis": "AtomGPT JARVIS",
    "cdvae_benchmark_alex": "CDVAE Alexandria",
    "cdvae_benchmark_jarvis": "CDVAE JARVIS",
    "flowmm_benchmark_alex": "FlowMM Alexandria",
    "flowmm_benchmark_jarvis": "FlowMM JARVIS"
}
plt.suptitle(bnchmk_name_dict.get(Path.cwd().name, ""), fontsize=30)
plt.subplots_adjust(top=0.88)
plt.savefig("distribution.png", format='png')
plt.close()

# ── Write metrics.json ─────────────────────────────────────────────────────
metrics = {
    "n_same_composition": len(comp_same),
    "n_total_composition": len(comp_total),
    "n_same_spacegroup": len(spg_same),
    "n_total_spacegroup": len(spg_total),
    "KLD": {
        "average_lat_params": avg_kld,
        "a": kl_divergence(x_a,y_a), "b": kl_divergence(x_b,y_b),
        "c": kl_divergence(x_c,y_c), "alpha": kl_divergence(x_alpha,y_alpha),
        "beta": kl_divergence(x_beta,y_beta), "gamma": kl_divergence(x_gamma,y_gamma),
        "lat": kl_divergence(x_lat_counts, y_lat_counts),
        "spg": kl_divergence(x_spg, y_spg),
        "weight": kl_divergence(x_Z, y_Z)
    },
    "EMD": {
        "a": emd_distance(x_a,y_a), "b": emd_distance(x_b,y_b),
        "c": emd_distance(x_c,y_c), "gamma": emd_distance(x_gamma,y_gamma),
        "lat": emd_distance(x_lat_counts, y_lat_counts),
        "spg": emd_distance(x_spg, y_spg),
        "weight": emd_distance(x_Z, y_Z)
    },
    "MAE": {
        "average_mae": avg_mae,
        "a": mean_absolute_error(x_a,y_a), "b": mean_absolute_error(x_b,y_b),
        "c": mean_absolute_error(x_c,y_c), "alpha": mean_absolute_error(x_alpha,y_alpha),
        "beta": mean_absolute_error(x_beta,y_beta), "gamma": mean_absolute_error(x_gamma,y_gamma)
    },
    "ranges": {
        "a": [min(x_a), max(x_a)], "b": [min(x_b), max(x_b)],
        "c": [min(x_c), max(x_c)], "gamma": [min(x_gamma), max(x_gamma)],
        "spg": [min(x_spg), max(x_spg)], "weight": [min(x_Z), max(x_Z)]
    }
}
with open(Path.cwd()/"metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=2)

print(f"✓ metrics written to {(Path.cwd()/'metrics.json').resolve()}")


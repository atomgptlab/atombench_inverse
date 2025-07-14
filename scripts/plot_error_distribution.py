import os
import json
from pathlib import Path
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.special import rel_entr
import statistics
from scipy.stats import wasserstein_distance
from scipy import stats
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.analysis.structure.spacegroup import Spacegroup3D
import pandas as pd
from jarvis.io.vasp.inputs import Poscar

# ── Matplotlib defaults ────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({"font.size": 18})

# ── Figure / GridSpec ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
the_grid = GridSpec(2, 3)

# ── Helper functions ──────────────────────────────────────────────────────
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
    return stats.entropy(p, q)

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("AI-AtomGen-prop-dft_3d-test-rmse.csv")
records = []
for _, row in df.iterrows():
    rec = {
        "target":    Poscar.from_string(row['target'].replace("\\n", "\n")).atoms.to_dict(),
        "predicted": Poscar.from_string(row['prediction'].replace("\\n", "\n")).atoms.to_dict()
    }
    records.append(rec)

# ── Extract targets / predictions for lattice params & angles ─────────────
x_a, y_a = [r["target"]["abc"][0]   for r in records], [r["predicted"]["abc"][0]   for r in records]
x_b, y_b = [r["target"]["abc"][1]   for r in records], [r["predicted"]["abc"][1]   for r in records]
x_c, y_c = [r["target"]["abc"][2]   for r in records], [r["predicted"]["abc"][2]   for r in records]
x_alpha, y_alpha = [r["target"]["angles"][0] for r in records], [r["predicted"]["angles"][0] for r in records]
x_beta,  y_beta  = [r["target"]["angles"][1] for r in records], [r["predicted"]["angles"][1] for r in records]
x_gamma, y_gamma = [r["target"]["angles"][2] for r in records], [r["predicted"]["angles"][2] for r in records]

# ── Histogram helper (avoid repetition) ───────────────────────────────────
def overlay_hist(ax, x, y, bins, xlabel, title):
    w_x = np.ones_like(x) / len(x) * 100
    w_y = np.ones_like(y) / len(y) * 100
    ax.hist(x, bins=bins, weights=w_x, alpha=0.6, color="tab:blue", label="target")
    ax.hist(y, bins=bins, weights=w_y, alpha=0.6, color="plum",    label="predicted")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax

# ── (a) a ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 0]),
             x_a, y_a,
             bins=np.arange(2, 7, 0.1),
             xlabel=r"a ($\AA$)",
             title="(a)").set_ylabel("Materials dist.")
plt.legend()

# ── (b) c ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 1]),
             x_c, y_c,
             bins=np.arange(2, 7, 0.1),
             xlabel=r"c ($\AA$)",
             title="(b)")

# ── (c) γ ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 2]),
             x_gamma, y_gamma,
             bins=np.arange(30, 150, 10),
             xlabel=r"$\gamma$ ($^\circ$)",
             title="(c)")

# ── Prepare composition / lattice system / spacegroup info ───────────────
comp, spg = [], []
x_spg, y_spg, x_Z, y_Z = [], [], [], []
x_lat, y_lat = [], []

for rec in records:
    a1, a2 = Atoms.from_dict(rec["target"]), Atoms.from_dict(rec["predicted"])
    x_Z.append(a1.composition.weight)
    y_Z.append(a2.composition.weight)

    # Crystal system
    try:
        lat_1 = Spacegroup3D(a1).crystal_system
    except Exception:
        lat_1 = None
    try:
        lat_2 = Spacegroup3D(a2).crystal_system
    except Exception:
        lat_2 = None
    x_lat.append(lat_1)
    y_lat.append(lat_2)

    # Spacegroup numbers
    try:
        sga1 = SpacegroupAnalyzer(a1.pymatgen_converter(), symprec=0.1)
        sga2 = SpacegroupAnalyzer(a2.pymatgen_converter(), symprec=0.1)
        x_spg.append(sga1.get_space_group_number())
        y_spg.append(sga2.get_space_group_number())
    except Exception:
        pass

# ── (d) **Spacegroup** – now using IDENTICAL bins for perfect overlay ────
ax_spg = plt.subplot(the_grid[1, 0])
bins_spg = np.arange(1, 231, 10)           # identical bin edges
overlay_hist(ax_spg,
             x_spg, y_spg,
             bins=bins_spg,
             xlabel="Spacegroup number",
             title="(d)").set_ylabel("Materials dist.")

# ── (e) Bravais lattice counts ───────────────────────────────────────────
lat_order = ["triclinic", "monoclinic", "orthorhombic",
             "tetragonal", "trigonal", "hexagonal", "cubic"]
lat_to_idx = {name: i for i, name in enumerate(lat_order)}
valid_lat = [(lx, ly) for lx, ly in zip(x_lat, y_lat) if lx and ly]
if valid_lat:
    x_lat, y_lat = zip(*valid_lat)
else:
    x_lat, y_lat = [], []
x_lat_counts = np.bincount([lat_to_idx[l] for l in x_lat], minlength=len(lat_order))
y_lat_counts = np.bincount([lat_to_idx[l] for l in y_lat], minlength=len(lat_order))

ax_lat = plt.subplot(the_grid[1, 1])
bar_w = 0.4
pos = np.arange(len(lat_order))

# overlay bars at the same positions
ax_lat.bar(pos, x_lat_counts,
           width=bar_w,
           alpha=0.6,
           label="target",
           color="tab:blue")
ax_lat.bar(pos, y_lat_counts,
           width=bar_w,
           alpha=0.6,
           label="predicted",
           color="plum")

# ticks now numbered 1–7 instead of names
ax_lat.set_xticks(pos)
ax_lat.set_xticklabels((pos + 1).tolist(), rotation=0, ha="center")
ax_lat.set_xlabel("Bravais lattice number")
ax_lat.set_title("(e)")


# ── (f) Molecular weight ────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[1, 2]),
             x_Z, y_Z,
             bins=np.arange(15, 2000, 100),
             xlabel="Weight (AMU)",
             title="(f)")

# ── Final layout & save ──────────────────────────────────────────────────
plt.tight_layout()
bench_lookup = {
    "agpt_benchmark_alex":  "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":"AtomGPT JARVIS",
    "cdvae_benchmark_alex": "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":"CDVAE JARVIS",
    "flowmm_benchmark_alex":"FlowMM Alexandria",
    "flowmm_benchmark_jarvis":"FlowMM JARVIS"
}
fig.subplots_adjust(top=0.88)
plt.suptitle(bench_lookup.get(Path.cwd().parts[-1], Path.cwd().name),
             fontsize=30)

out_png = f"{Path.cwd().name}_distribution.png"
plt.savefig(out_png, format="png")
plt.close()
print(f"✓ saved {out_png}")

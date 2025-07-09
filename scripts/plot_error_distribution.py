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
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.family']    = 'serif'

from matplotlib.gridspec import GridSpec
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.analysis.structure.spacegroup import Spacegroup3D
import pandas as pd
from jarvis.io.vasp.inputs import Poscar

the_grid = GridSpec(2, 3)
plt.rcParams.update({"font.size": 18})
fig = plt.figure(figsize=(14, 8))
mpl.rcParams['font.family']    = 'serif'

def emd_distance(p, q, bins=None):
    """
    Compute Earth Mover's Distance (Wasserstein distance) between two 1D counts.

    Args:
        p (np.ndarray): First count or histogram (can be values or weights).
        q (np.ndarray): Second count or histogram.
        bins (np.ndarray or list, optional): Positions of each bin (same length as p and q).
                                             If None, assumes equally spaced bins [0, 1, ..., N-1].

    Returns:
        float: Earth Mover's Distance (EMD)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize if they are not already
    p /= np.sum(p)
    q /= np.sum(q)

    # Use default positions if not specified
    if bins is None:
        bins = np.arange(len(p))

    return wasserstein_distance(bins, bins, u_weights=p, v_weights=q)


def kl_divergence(p, q):
    """
    Compute KL divergence between two counts (p || q)
    Args:
        p (np.array): First count (usually the true distribution)
        q (np.array): Second count (usually the predicted distribution)
    Returns:
        float: KL divergence
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize to make sure they are probability counts
    p /= np.sum(p)
    q /= np.sum(q)
    return stats.entropy(p, q)
    # Use scipy's rel_entr (returns elementwise p * log(p/q))
    # return np.sum(rel_entr(p, q))


df = pd.read_csv("AI-AtomGen-prop-dft_3d-test-rmse.csv")
d = []
for ii, i in df.iterrows():
    info = {}
    info['target'] = Poscar.from_string(i['target'].replace("\\n", "\n")).atoms.to_dict()
    info['predicted'] = Poscar.from_string(i['prediction'].replace("\\n", "\n")).atoms.to_dict()
    d.append(info)


x_a = [i["target"]["abc"][0] for i in d]
y_a = [i["predicted"]["abc"][0] for i in d]
x_b = [i["target"]["abc"][1] for i in d]
y_b = [i["predicted"]["abc"][1] for i in d]
x_c = [i["target"]["abc"][2] for i in d]
y_c = [i["predicted"]["abc"][2] for i in d]
x_alpha = [i["target"]["angles"][0] for i in d]
y_alpha = [i["predicted"]["angles"][0] for i in d]
x_beta = [i["target"]["angles"][1] for i in d]
y_beta = [i["predicted"]["angles"][1] for i in d]
x_gamma = [i["target"]["angles"][2] for i in d]
y_gamma = [i["predicted"]["angles"][2] for i in d]

plt.subplot(the_grid[0, 0])
weights_x = np.ones_like(x_a) / len(x_a) * 100
weights_y = np.ones_like(y_a) / len(y_a) * 100
plt.hist(
    x_a,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="target",
    alpha=0.6,
    color="tab:blue",
)
plt.hist(
    y_a,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="predicted",
    color="plum",
    alpha=0.6,
)
plt.xlabel(r"a ($\AA$)")
plt.title("(a)")
plt.ylabel("Materials dist.")
plt.legend()

plt.subplot(the_grid[0, 1])
weights_x = np.ones_like(x_b) / len(x_b) * 100
weights_y = np.ones_like(y_b) / len(y_b) * 100
plt.hist(
    x_b,
    bins=np.arange(2, 7, 0.1),
    weights=weights_x,
    label="target_b",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_b,
    bins=np.arange(2, 7, 0.1),
    weights=weights_y,
    label="predicted_b",
    color="plum",
    alpha=0.6,
)
plt.xlabel(r"b ($\AA$)")
plt.title("(b)")

plt.subplot(the_grid[0, 2])
weights_x = np.ones_like(x_gamma) / len(x_gamma) * 100
weights_y = np.ones_like(y_gamma) / len(y_gamma) * 100
plt.hist(
    x_gamma,
    bins=np.arange(30, 150, 10),
    weights=weights_x,
    label="target_gamma",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_gamma,
    bins=np.arange(30, 150, 10),
    weights=weights_y,
    label="predicted_gamma",
    color="plum",
    alpha=0.6,
)
plt.xlabel(r"$\gamma$ ($^\circ$)")
plt.title("(c)")

comp = []
spg = []
samps_spg = []
samps_comp = []
x_spg = []
y_spg = []
x_Z = []
y_Z = []
x_lat = []
y_lat = []
for i in d:
    a1 = Atoms.from_dict(i["target"])
    a2 = Atoms.from_dict(i["predicted"])
    comp1 = a1.composition.reduced_formula
    comp2 = a2.composition.reduced_formula
    x_Z.append(a1.composition.weight)
    y_Z.append(a2.composition.weight)
    lat_1 = Spacegroup3D(a1).crystal_system
    print(lat_1)
    lat_2 = Spacegroup3D(a2).crystal_system

    x_lat.append(lat_1)
    y_lat.append(lat_2)

    if comp1 == comp2:
        comp.append(i)
    else:
        print("different comp", comp1, comp2)
    samps_comp.append(i)
    try:
        sga1 = SpacegroupAnalyzer(a1.pymatgen_converter(), symprec=0.1)
        sga2 = SpacegroupAnalyzer(a2.pymatgen_converter(), symprec=0.1)
        spg1 = sga1.get_space_group_number()
        spg2 = sga2.get_space_group_number()
        x_spg.append(spg1)
        y_spg.append(spg2)
        if spg1 == spg2:
            spg.append(i)
        samps_spg.append(i)
        if spg1 == spg2 and comp1 == comp2:
            matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
            rms_dist = matcher.get_rms_anonymous(
                a1.pymatgen_converter(), a2.pymatgen_converter()
            )
            print(a1)
            print(a2)
            print(rms_dist)
            print(i["id"], spg1, spg2)
    except:
        pass

# ── Encode Bravais lattice categories → numerical counts ────────────────
lat_order = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
]
lat_to_idx = {name: i for i, name in enumerate(lat_order)}

x_lat_idx = np.array([lat_to_idx[l] for l in x_lat], dtype=int)
y_lat_idx = np.array([lat_to_idx[l] for l in y_lat], dtype=int)

x_lat_counts = np.bincount(x_lat_idx, minlength=len(lat_order))
y_lat_counts = np.bincount(y_lat_idx, minlength=len(lat_order))

average_lattice_params_kld = statistics.mean([
    kl_divergence(x_a, y_a),
    kl_divergence(x_b, y_b),
    kl_divergence(x_c, y_c),
    kl_divergence(x_alpha, y_alpha),
    kl_divergence(x_beta, y_beta),
    kl_divergence(x_gamma, y_gamma),
])

average_mae = statistics.mean([
    mean_absolute_error(x_a, y_a),
    mean_absolute_error(x_b, y_b),
    mean_absolute_error(x_c, y_c),
    mean_absolute_error(x_alpha, y_alpha),
    mean_absolute_error(x_beta, y_beta),
    mean_absolute_error(x_gamma, y_gamma)
])

print("comp", len(comp), len(samps_comp), "spg", len(spg), len(samps_spg))
print("average lattice params KLD", average_lattice_params_kld)
print("KLD a", kl_divergence(x_a, y_a))
print("KLD b", kl_divergence(x_b, y_b))
print("KLD c", kl_divergence(x_c, y_c))
print("KLD alpha", kl_divergence(x_alpha, y_alpha))
print("KLD beta", kl_divergence(x_beta, y_beta))
print("KLD gamma", kl_divergence(x_gamma, y_gamma))
print("KLD lat", kl_divergence(x_lat_counts, y_lat_counts))
print("KLD spg", kl_divergence(x_spg, y_spg))
print("EMD a", emd_distance(x_a, y_a))
print("EMD b", emd_distance(x_b, y_b))
print("EMD gamma", emd_distance(x_gamma, y_gamma))
print("EMD spg", emd_distance(x_spg, y_spg))
print("EMD lat", emd_distance(x_lat_counts, y_lat_counts))
print("Min Max a", min(x_a), max(x_a))
print("Min Max b", min(x_b), max(x_b))
print("Min Max c", min(x_c), max(x_c))
print("Min Max gamma", min(x_gamma), max(x_gamma))

plt.subplot(the_grid[1, 2])
print("xZ_", min(x_Z), max(x_Z))
weights_x = np.ones_like(x_Z) / len(x_Z) * 100
weights_y = np.ones_like(y_Z) / len(y_Z) * 100
plt.hist(
    x_Z,
    bins=np.arange(15, 2000, 100),
    weights=weights_x,
    label="target_wt",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_Z,
    bins=np.arange(15, 2000, 100),
    weights=weights_y,
    label="predicted_wt",
    color="plum",
    alpha=0.6,
)
plt.xlabel("Weight (AMU)")
plt.title("(f)")

plt.subplot(the_grid[1, 0])
weights_x = np.ones_like(x_spg) / len(x_spg) * 100
weights_y = np.ones_like(y_spg) / len(y_spg) * 100
plt.hist(
    x_spg,
    bins=np.arange(0, 231, 10),
    weights=weights_x,
    label="target_spg",
    color="tab:blue",
    alpha=0.6,
)
plt.hist(
    y_spg,
    bins=np.arange(0, 220, 10),
    weights=weights_y,
    label="predicted_spg",
    color="plum",
    alpha=0.6,
)
plt.ylabel("Materials dist.")
plt.xlabel("Spacegroup number")
plt.title("(d)")

plt.subplot(the_grid[1, 1])
bar_w = 0.35
pos    = np.arange(len(lat_order))
plt.bar(
    pos - bar_w/2,
    x_lat_counts,
    bar_w,
    label="target_lat",
    color="tab:blue",
    alpha=0.6,
)
plt.bar(
    pos + bar_w/2,
    y_lat_counts,
    bar_w,
    label="predicted_lat",
    color="plum",
    alpha=0.6,
)
plt.xticks(pos, lat_order, rotation=45, ha="right")
plt.ylabel("Materials count")
plt.xlabel("Bravais lattice")
plt.title("(e)")

plt.tight_layout()
bnchmk_name_dict = {
    "agpt_benchmark_alex": "AtomGPT Alexandria",
    "agpt_benchmark_jarvis": "AtomGPT JARVIS",
    "cdvae_benchmark_alex": "CDVAE Alexandria",
    "cdvae_benchmark_jarvis": "CDVAE JARVIS",
    "flowmm_benchmark_alex": "FlowMM Alexandria",
    "flowmm_benchmark_jarvis": "FlowMM JARVIS"
}
fig.subplots_adjust(top=0.88)
plt.suptitle(bnchmk_name_dict[Path(os.getcwd()).parts[-1]],
             fontsize=30
)
plt.savefig("distribution.png", format='png')
plt.close()

print("Min Max spg", min(x_spg), max(x_spg))
x_Z = np.array(x_Z)
y_Z = np.array(y_Z)
mask = np.isfinite(x_Z) & np.isfinite(y_Z)
x_Z = x_Z[mask]
y_Z = y_Z[mask]
print("Min Max weight", min(x_Z), max(x_Z))
print("KLD weight", kl_divergence(x_Z, y_Z))
print("EMD weight", emd_distance(x_Z, y_Z))
print("a", mean_absolute_error(x_a, y_a))
print("b", mean_absolute_error(x_b, y_b))
print("c", mean_absolute_error(x_c, y_c))
print("alpha", mean_absolute_error(x_alpha, y_alpha))
print("beta", mean_absolute_error(x_beta, y_beta))
print("gamma", mean_absolute_error(x_gamma, y_gamma))

# Load predictions
# d = loadjson('dft_2d_formula_based_dft_2d_unsloth/.../test_predictions.json')

# Prepare data arrays
labels = ["a", "b", "c", "alpha", "beta", "gamma"]
targets = [
    np.array(
        [
            i["target"]["abc"][j] if j < 3 else i["target"]["angles"][j - 3]
            for i in d
        ]
    )
    for j in range(6)
]
preds = [
    np.array(
        [
            (
                i["predicted"]["abc"][j]
                if j < 3
                else i["predicted"]["angles"][j - 3]
            )
            for i in d
        ]
    )
    for j in range(6)
]
bins = [
    np.arange(2, 7, 0.1),
    np.arange(2, 7, 0.1),
    np.arange(10, 30, 0.1),
    np.arange(0, 180, 10),
] * 2

# Plot setup
plt.rcParams.update({"font.size": 12})
fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 3, figure=fig)

for idx, label in enumerate(labels):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.hist(
        targets[idx],
        bins=bins[idx],
        label=f"Target {label}",
        alpha=0.6,
        color="tab:blue",
    )
    ax.hist(
        preds[idx],
        bins=bins[idx],
        label=f"Predicted {label}",
        alpha=0.6,
        color="plum",
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Material Count")
    ax.set_title(f"Distribution of {label}")
    ax.legend()

fig.tight_layout()
# plt.savefig("raman_pretty_distribution_plot.png")
plt.close()

metrics = {
    "n_same_composition": len(comp),
    "n_total_composition": len(samps_comp),
    "n_same_spacegroup": len(spg),
    "n_total_spacegroup": len(samps_spg),

    "KLD": {
        "average_lat_params": average_lattice_params_kld,
        "a": kl_divergence(x_a, y_a),
        "b": kl_divergence(x_b, y_b),
        "c": kl_divergence(x_c, y_c),
        "alpha": kl_divergence(x_alpha, y_alpha),
        "beta": kl_divergence(x_beta, y_beta),
        "gamma": kl_divergence(x_gamma, y_gamma),
        "lat": kl_divergence(x_lat_counts, y_lat_counts),
        "spg": kl_divergence(x_spg, y_spg),
        "weight": kl_divergence(x_Z, y_Z),
    },

    "EMD": {
        "a": emd_distance(x_a, y_a),
        "b": emd_distance(x_b, y_b),
        "gamma": emd_distance(x_gamma, y_gamma),
        "spg": emd_distance(x_spg, y_spg),
        "lat": emd_distance(x_lat_counts, y_lat_counts),
        "weight": emd_distance(x_Z, y_Z),
    },

    "MAE": {
        "average_mae": average_mae,
        "a": mean_absolute_error(x_a, y_a),
        "b": mean_absolute_error(x_b, y_b),
        "c": mean_absolute_error(x_c, y_c),
        "alpha": mean_absolute_error(x_alpha, y_alpha),
        "beta": mean_absolute_error(x_beta, y_beta),
        "gamma": mean_absolute_error(x_gamma, y_gamma),
    },

    "ranges": {
        "a": [min(x_a), max(x_a)],
        "b": [min(x_b), max(x_b)],
        "c": [min(x_c), max(x_c)],
        "gamma": [min(x_gamma), max(x_gamma)],
        "spg": [min(x_spg), max(x_spg)],
        "weight": [min(x_Z), max(x_Z)],
    }
}

run_dir = Path.cwd()
json_path = run_dir / "metrics.json"
with open(json_path, "w") as fp:
    json.dump(metrics, fp, indent=2)

print(f"✓ metrics written to {json_path.resolve()}")


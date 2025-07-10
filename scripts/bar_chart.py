import json
import os
from pathlib import Path
import sys

# Force a non‐interactive backend in case you’re headless
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.family']    = 'serif'

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

ROOT = Path.cwd()  # where we’re running from
print(f"DEBUG: Running script in {ROOT}", file=sys.stderr)

rows = []

for subdir in sorted(ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    metrics_fp = subdir / "metrics.json"
    if not metrics_fp.is_file():
        print(f"⚠️  no metrics.json in {subdir.name} – skipped", file=sys.stderr)
        continue

    with metrics_fp.open() as fh:
        rec = json.load(fh)

    rec.setdefault("benchmark_name", subdir.name)
    flat = pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict()
    rows.append(flat)
    print(f"DEBUG: Loaded metrics for {rec['benchmark_name']}", file=sys.stderr)

if not rows:
    print("ERROR: No metrics.json files found – exiting", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)
print("DEBUG: DataFrame shape:", df.shape, file=sys.stderr)
print("DEBUG: DataFrame columns:", df.columns.tolist(), file=sys.stderr)

bnchmk_name_dict = {
    "agpt_benchmark_alex": "AtomGPT Alexandria",
    "agpt_benchmark_jarvis": "AtomGPT JARVIS",
    "cdvae_benchmark_alex": "CDVAE Alexandria",
    "cdvae_benchmark_jarvis": "CDVAE JARVIS",
    "flowmm_benchmark_alex": "FlowMM Alexandria",
    "flowmm_benchmark_jarvis": "FlowMM JARVIS"
}

col_map = {
    'KLD.a': r'$a$',
    'KLD.b': r'$b$',
    'KLD.c': r'$c$',
    'KLD.alpha': r'$\alpha$',
    'KLD.beta': r'$\beta$',
    'KLD.gamma': r'$\gamma$'
}

if "benchmark_name" in df.columns:
    cols = ["benchmark_name"] + [c for c in df.columns if c != "benchmark_name"]
    df = df[cols]
else:
    print("ERROR: 'benchmark_name' column missing", file=sys.stderr)
    sys.exit(1)

params = ['KLD.a','KLD.b','KLD.c','KLD.alpha','KLD.beta','KLD.gamma']
missing = [p for p in params if p not in df.columns]
if missing:
    print("ERROR: Missing metric columns:", missing, file=sys.stderr)
    sys.exit(1)

plot_df = df.set_index('benchmark_name')[params].rename(index=bnchmk_name_dict).rename(columns=col_map)
print("DEBUG: plot_df shape:", plot_df.shape, file=sys.stderr)
print("DEBUG: plot_df head:\n", plot_df.head(), file=sys.stderr)

ax = plot_df.plot(
    kind='bar',
    figsize=(10, 8),
    edgecolor='k'
)

ax.set_xlabel('', fontsize=16)
ax.set_ylabel('KL Divergence (Nats)', fontsize=16)
ax.set_title('KL Divergence of Predicted vs. \n Target Lattice Parameter Distributions',
             fontsize=22)
ax.legend(title='Lattice Parameter',
          title_fontsize=15,
          fontsize=15
)

plt.xticks(rotation=30, ha='right', fontsize=13)
plt.yticks(fontsize=15)
plt.tight_layout()

out_path = ROOT / 'comparison_bar_chart.png'
print(f"DEBUG: Saving figure to {out_path}", file=sys.stderr)
plt.savefig(out_path, dpi=300)
# plt.show()  # removed to avoid hanging
print("DEBUG: Done.", file=sys.stderr)

#!/usr/bin/env python3
"""Vẽ line chart các metric theo AF bin từ report đã chấm."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vn1k_khv/summary_by_af_bin.csv"
OUT = ROOT / "reports/vn1k_khv/Figure_metrics_by_af.svg"
METHODS = ["AutoHLA", "SNP2HLA", "CookHLA", "HIBAG", "DEEP*HLA"]
BINS = ["<1%", "1-5%", "5-10%", "10-20%", ">=20%"]
METRICS = [("f1", "F1"), ("sn", "Sensitivity"), ("ppv", "PPV"),
           ("concordance", "Concordance")]

df = pd.read_csv(REPORT)
df = df[df.arm.isin(METHODS)].copy()
df["maf_bin"] = pd.Categorical(df.maf_bin, BINS, ordered=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, (column, title) in zip(axes.flat, METRICS):
    for method in METHODS:
        part = df[df.arm.eq(method)].sort_values("maf_bin")
        ax.plot(part.maf_bin.astype(str), part[column], marker="o", label=method)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.set_xlabel("AF bin")
    ax.set_ylabel(column)
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("VN1K → KHV: AutoHLA và baseline theo AF bin")
fig.savefig(OUT, facecolor="white")
print(OUT)

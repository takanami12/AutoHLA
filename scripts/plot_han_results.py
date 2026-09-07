#!/usr/bin/env python3
"""Vẽ line chart các metric theo AF bin cho hai protocol HAN."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BINS = ["<1%", "1-5%", "5-10%", "10-20%", ">=20%"]
CONFIG = {
    "han_cv": {
        "csv": ROOT / "reports/han_cv/summary_by_af_bin.csv",
        "out": ROOT / "reports/han_cv/Figure_metrics_by_af.svg",
        "title": "HAN 10-fold CV: AutoHLA và baseline theo AF bin",
        "methods": ["AutoHLA+pair+ridge", "AutoHLA", "AEHLA+pair+ridge",
                    "CookHLA", "SNP2HLA"],
    },
    "han_chb": {
        "csv": ROOT / "reports/han_chb/summary_by_af_bin.csv",
        "out": ROOT / "reports/han_chb/Figure_metrics_by_af.svg",
        "title": "HAN → CHB: AutoHLA và baseline theo AF bin",
        "methods": ["AutoHLA", "AutoHLA+pair", "SNP2HLA", "CookHLA", "DEEP*HLA"],
    },
}
METRICS = [("f1", "F1"), ("sn", "Sensitivity"), ("ppv", "PPV"),
           ("concordance", "Concordance")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(CONFIG), required=True)
    args = parser.parse_args()
    spec = CONFIG[args.experiment]
    frame = pd.read_csv(spec["csv"])
    frame = frame[frame.arm.isin(spec["methods"])].copy()
    frame["maf_bin"] = pd.Categorical(frame.maf_bin, BINS, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, METRICS):
        for method in spec["methods"]:
            part = frame[frame.arm.eq(method)].sort_values("maf_bin")
            if not part.empty:
                ax.plot(part.maf_bin.astype(str), part[column], marker="o", label=method)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xlabel("AF bin")
        ax.set_ylabel(column)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(spec["title"])
    spec["out"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(spec["out"], facecolor="white")
    print(spec["out"])


if __name__ == "__main__":
    main()

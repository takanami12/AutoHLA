#!/usr/bin/env python3
"""Tạo duy nhất một manifest CV dùng chung cho mọi method trên HAN."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCI = ("A", "B", "C", "DPB1", "DRB1", "DQA1", "DQB1")
SEEDS = (13, 29, 43, 59, 71, 77, 89, 101, 127, 149)


def assign(ids, labels, k, seed):
    """Greedy rare-first multilabel stratification, deterministic for one seed."""
    rng = random.Random(seed)
    tokens = {sid: tuple(labels.loc[sid, [f"{g}_{h}" for g in LOCI for h in (1, 2)]])
              for sid in ids}
    freq = Counter(x for values in tokens.values() for x in values)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda sid: sum(1 / freq[x] for x in tokens[sid]), reverse=True)
    folds, counts = [[] for _ in range(k)], [Counter() for _ in range(k)]
    target = len(ids) / k
    for sid in shuffled:
        score = [sum(counts[f][x] / freq[x] for x in tokens[sid]) + len(folds[f]) / target
                 for f in range(k)]
        chosen = min(range(k), key=lambda f: (score[f], len(folds[f]), f))
        folds[chosen].append(sid)
        counts[chosen].update(tokens[sid])
    return folds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=77, help="Seed cố định cho manifest, không phải model")
    p.add_argument("--out", type=Path, default=ROOT / "results/cv_han/protocol")
    args = p.parse_args()

    labels = pd.read_csv(ROOT / "HAN_dataset/HAN.HLA.4digit.tsv", sep="\t", dtype=str).set_index("sample_id")
    labels = labels[[f"{g}_{h}" for g in LOCI for h in (1, 2)]]
    outer = assign(labels.index.tolist(), labels, args.folds, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, test in enumerate(outer, 1):
        pool = labels.index.difference(test, sort=False).tolist()
        # 5% of the 9-fold training pool is validation; test remains untouched.
        val = assign(pool, labels, 20, args.seed + fold)[0]
        val_set, test_set = set(val), set(test)
        split = pd.Series("train", index=labels.index, name="split")
        split.loc[list(val_set)] = "val"
        split.loc[list(test_set)] = "test"
        fold_dir = args.out / f"fold{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        for name in ("train", "val", "test"):
            ids = split.index[split.eq(name)]
            (fold_dir / f"{name}.samples").write_text("\n".join(ids) + "\n")
            labels.loc[ids].to_csv(fold_dir / f"{name}.labels.tsv", sep="\t", index_label="sample_id")
        rows.extend({"sample_id": sid, "fold": fold, "split": value}
                    for sid, value in split.items())

    manifest = pd.DataFrame(rows)
    assert manifest.groupby("sample_id").split.apply(lambda x: (x == "test").sum()).eq(1).all()
    assert manifest.groupby("fold").sample_id.nunique().eq(len(labels)).all()
    manifest.to_csv(args.out / "manifest.tsv", sep="\t", index=False)
    metadata = {
        "dataset": "HAN", "assembly": "hg18/GRCh36", "outer_folds": args.folds,
        "split_seed": args.seed, "model_seeds": SEEDS, "loci": LOCI,
        "n_samples": len(labels), "n_snps": 25996,
        "maf_source": "outer-train reference haplotypes of each fold",
        "metrics": ["sn", "ppv", "f1", "concordance", "r2"],
        "validation_fraction_of_train_pool": 0.05,
        "af_bins": ["<1%", "1-5%", "5-10%", "10-20%", ">=20%"],
        "rare_af_bins": ["AF=0", "0-0.1%", "0.1-0.5%", "0.5-1%", "1-5%", "5-10%", "10-20%", ">=20%"],
    }
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(manifest.groupby(["fold", "split"]).size().unstack().to_string())


if __name__ == "__main__":
    main()

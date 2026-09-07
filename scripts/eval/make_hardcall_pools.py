#!/usr/bin/env python3
"""Convert cac file dosage mem da pool (results/cv_vn1k/predictions/*.dosage.csv.gz)
sang hard-call, dung luat chung CLAUDE.md: "Cac phuong phap tra ve dosage mem deu se
duoc convert het sang hard-call allele".

Hard-call moi (sample, gene): 2 allele co dosage-gop cao nhat (dong hop neu chi co
1 allele quan sat). Ghi ra {method}_hardcall.dosage.csv.gz voi dosage la so ban sao
hard 0/1/2, cung schema de cv_pool.py score() dung lai duoc nguyen.
"""
import argparse
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / os.environ.get("HARDCALL_PRED", "results/cv_vn1k/predictions")
METHODS = ["SNP2HLA", "CookHLA", "DeepHLA", "HIBAG", "flohla", "AEHLA_PAIR", "base"]
# HARDCALL_METHODS THAY HAN danh sach (cohort khac co bo method khac);
# HARDCALL_EXTRA chi BO SUNG vao danh sach dang co.
_replace = [m for m in os.environ.get("HARDCALL_METHODS", "").split(",") if m]
if _replace:
    METHODS = _replace
METHODS += [m for m in os.environ.get("HARDCALL_EXTRA", "").split(",") if m]
# Ap len file DA la hard-call cung khong sao: top-2 cua mot phan bo 1/1 hoac 2
# tra lai chinh no. Chay tat ca qua cung mot ham la cach dam bao luat dong nhat
# BANG CAU TRUC chu khong bang loi hua.


def hardcall(df):
    rows = []
    for (sid, fold, gene), sub in df.groupby(["sample_id", "fold", "gene"]):
        agg = sub.groupby("allele")["dosage"].sum().sort_values(ascending=False)
        alleles = list(agg.index)
        a1 = alleles[0]
        a2 = alleles[1] if len(alleles) > 1 else alleles[0]
        for a in (a1, a2):
            rows.append((sid, fold, gene, a))
    hc = pd.DataFrame(rows, columns=["sample_id", "fold", "gene", "allele"])
    hc["dosage"] = 1
    return hc.groupby(["sample_id", "fold", "gene", "allele"], as_index=False)["dosage"].sum()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        help="tên stem trong thư mục predictions/")
    args = parser.parse_args(argv)
    for method in args.methods:
        src = PRED / f"{method}.dosage.csv.gz"
        df = pd.read_csv(src)
        hc = hardcall(df)
        dest = PRED / f"{method}_hardcall.dosage.csv.gz"
        hc.to_csv(dest, index=False)
        print(f"{method}: {len(df)} dong mem -> {len(hc)} dong hard-call -> {dest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gop calls.csv cua 40 o mot arm -> mot file dosage cho evaluator chuan cua repo.

`autohla impute` ghi mot dong moi (mau x gene) voi allele_1/allele_2 la goi CUNG.
scripts/eval/cv10_canonical_metrics.py doc dinh dang khac: mot dong moi
(mau, fold, gene, allele) voi dosage 0/1/2. Chuyen doi la mot phep dem, khong
phai mot phep quyet dinh -- dong hop thi mot dong dosage=2, di hop thi hai dong
dosage=1.

    python tools/calls_to_dosage.py --sweep results/autohla_cv_vn1k \
        --arm cm_real --name AUTOHLA_CM_REAL
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Cohort khac co thu muc du doan rieng, va evaluator can MOI method nam chung mot
# cho de bang AF va allele universe trung nhau tuyet doi.
PRED = ROOT / os.environ.get("POOL_PRED", "results/cv_vn1k/predictions")


def pool(sweep, arm, folds=range(1, 11), groups=(1, 2, 3, 4)):
    rows, missing = [], []
    for fold in folds:
        for group in groups:
            path = Path(sweep) / arm / f"fold{fold:02d}_g{group}" / "calls.csv"
            if not path.exists():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path, dtype={"sample_id": str})
            for record in frame.itertuples(index=False):
                counts = Counter([record.allele_1, record.allele_2])
                for allele, dosage in counts.items():
                    rows.append((record.sample_id, fold, record.gene, allele, dosage))
    return pd.DataFrame(rows, columns=["sample_id", "fold", "gene", "allele",
                                       "dosage"]), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True, help="ten file trong predictions/")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    lo, _, hi = args.folds.partition("-")
    folds = range(int(lo), int(hi or lo) + 1)
    frame, missing = pool(args.sweep, args.arm, folds=folds)
    if missing and not args.allow_missing:
        raise SystemExit("thieu {} o:\n  {}".format(len(missing),
                                                    "\n  ".join(missing[:5])))
    if missing:
        print(f"CANH BAO: thieu {len(missing)} o, van ghi", file=sys.stderr)
    out = PRED / f"{args.name}.dosage.csv.gz"
    frame.to_csv(out, index=False)
    print(f"{len(frame)} dong, {frame.sample_id.nunique()} mau, "
          f"{frame.gene.nunique()} gene -> {out}")
    # Kiem re nhat cho mot bug am tham: moi (mau, gene) phai co dung 2 ban sao.
    per_cell = frame.groupby(["sample_id", "fold", "gene"]).dosage.sum()
    bad = per_cell[per_cell != 2]
    print(f"o co tong ban sao != 2: {len(bad)}" + (f"  VD {bad.head(3).to_dict()}"
                                                  if len(bad) else "  (dung)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

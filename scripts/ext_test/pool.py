#!/usr/bin/env python3
"""Gop du doan cua 5 method tren mot o external test -> dinh dang evaluator chuan.

Moi method ra mot file `do noi bo<cfg>/predictions/<Method>.dosage.csv.gz`
voi 4 cot `sample_id,fold,gene,allele,dosage` (fold luon la 91). Do la dinh dang
DUY NHAT ma `cv10_canonical_metrics` doc, nen ca 5 method di qua
cung mot allele universe / bang AF / luat goi cung.

    python pool vn1k_khv [--methods SNP2HLA HIBAG...]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from common import LOCI, code_to_2field, known_alleles, normalize_allele  # noqa: E402
from cv_pool import vcf_dosages  # noqa: E402

FOLD = 91
MIN_DOSAGE = 1e-3
COLUMNS = ["sample_id", "fold", "gene", "allele", "dosage"]

CONFIGS = {
    "vn1k_khv": dict(
        cohort="VN1K_KHV",
        proto=ROOT / "results/cv_vn1k/protocol/fold91",
        snp2hla=[ROOT / "results/cv_vn1k/baselines/fold91/snp2hla/fold91.bgl.vcf"],
        cookhla=ROOT / "results/cv_vn1k/baselines/fold91/cookhla",
    ),
    "han_chb": dict(
        cohort="HAN_CHB",
        proto=ROOT / "results/cv_han/protocol/fold91",
        snp2hla=ROOT / "results/cv_han/predictions/APMDA_CHB/SNP2HLA/seed77/fold91/dosage.csv",
        cookhla=ROOT / "results/cv_han/predictions/APMDA_CHB/CookHLA/seed77/fold91/dosage.csv",
    ),
}
for name, cfg in CONFIGS.items():
    cfg["out"] = ROOT / "results/ext_test" / name / "predictions"
    cfg["deephla"] = ROOT / "results/ext_test" / name / "deephla"
    cfg["hibag"] = ROOT / "results/ext_test" / name / "hibag"
    cfg["autohla"] = ROOT / "results/ext_test" / name / "autohla/base"


def rows_from_dosage_frame(path, test_ids, known):
    """dosage.csv cua adapter Han: hang = GENE_ALLELE, cot = mau."""
    frame = pd.read_csv(path, index_col=0)
    frame.columns = frame.columns.astype(str)
    rows = []
    for marker, series in frame.iterrows():
        gene, _, code = str(marker).partition("_")
        allele = normalize_allele(code_to_2field(code, known.get(gene)))
        if gene not in LOCI or not allele:
            continue
        for sid in test_ids:
            value = float(series.get(sid, 0.0))
            if value > MIN_DOSAGE:
                rows.append((sid, FOLD, gene, allele, round(value, 4)))
    return rows


def rows_from_vcf(paths, cook, test_ids, known):
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise SystemExit("chua co output VCF")
    rows = []
    for (gene, allele), series in vcf_dosages(paths, cook, known).items():
        values = series.reindex(test_ids)
        keep = values.notna() & (values > MIN_DOSAGE)
        rows.extend((sid, FOLD, gene, allele, round(float(v), 4))
                    for sid, v in values[keep].items())
    return rows


def rows_from_hibag(directory, test_ids):
    rows = []
    for gene in LOCI:
        frame = pd.read_csv(Path(directory) / f"Imputed_test_{gene}.csv", dtype=str)
        frame = frame.set_index("SampleID")
        for sid in test_ids:
            if sid not in frame.index:
                continue
            for column in ("Pred_H1", "Pred_H2"):
                allele = normalize_allele(frame.at[sid, column])
                if allele:
                    rows.append((sid, FOLD, gene, allele, 1.0))
    return rows


def rows_from_deephla(directory, name, test_ids, known):
    directory = Path(directory)
    fam = pd.read_csv(directory / "test.fam", sep=r"\s+", header=None, dtype=str)
    raw = pd.read_csv(directory / f"{name}.deephla.dosage", sep="\t", header=None,
                      index_col=0)
    values = raw.iloc[:, 2:]
    values.columns = fam[1].tolist()
    rows = []
    for marker, series in values.iterrows():
        marker = str(marker)
        if not marker.startswith("HLA_"):
            continue
        gene, _, code = marker[4:].partition("_")
        # Bundle vn1k viet 4 so co dau hai cham (01:01), bundle han viet lien (0101);
        # ca hai deu kem marker NHOM 2 so (HLA_A_01) phai bo, khong phai 2-field.
        if gene not in LOCI or len(code.replace(":", "")) < 4:
            continue
        allele = normalize_allele(code_to_2field(code, known.get(gene)))
        for sid in test_ids:
            value = float(series.get(sid, 0.0))
            if allele and value > MIN_DOSAGE:
                rows.append((sid, FOLD, gene, allele, round(value, 4)))
    return rows


def rows_from_autohla(directory, test_ids):
    rows = []
    for group in (1, 2, 3, 4):
        path = Path(directory) / f"fold{FOLD:02d}_g{group}" / "calls.csv"
        frame = pd.read_csv(path, dtype={"sample_id": str})
        for record in frame.itertuples(index=False):
            for allele, dosage in Counter([record.allele_1, record.allele_2]).items():
                rows.append((str(record.sample_id), FOLD, record.gene, allele, dosage))
    return rows


def build(name, methods, autohla_dir=None, autohla_name="AutoHLA"):
    cfg = CONFIGS[name]
    known = known_alleles(cfg["cohort"])
    test_ids = (cfg["proto"] / "test.samples").read_text().split()
    cfg["out"].mkdir(parents=True, exist_ok=True)
    for method in methods:
        if method == "SNP2HLA":
            source = cfg["snp2hla"]
            rows = (rows_from_dosage_frame(source, test_ids, known)
                    if isinstance(source, Path) else
                    rows_from_vcf(source, False, test_ids, known))
        elif method == "CookHLA":
            source = cfg["cookhla"]
            rows = (rows_from_dosage_frame(source, test_ids, known)
                    if source.suffix == ".csv" else
                    rows_from_vcf(sorted(source.glob("*raw_imputation_out.vcf")),
                                  True, test_ids, known))
        elif method == "HIBAG":
            rows = rows_from_hibag(cfg["hibag"], test_ids)
        elif method == "DeepHLA":
            rows = rows_from_deephla(cfg["deephla"], name, test_ids, known)
        elif method == "AutoHLA":
            rows = rows_from_autohla(autohla_dir or cfg["autohla"], test_ids)
        else:
            raise SystemExit(f"khong biet method {method}")
        frame = pd.DataFrame(rows, columns=COLUMNS)
        covered = frame.sample_id.nunique()
        if covered != len(test_ids):
            print(f"CANH BAO {method}: {covered}/{len(test_ids)} mau co du doan")
        out_name = autohla_name if method == "AutoHLA" else method
        destination = cfg["out"] / f"{out_name}.dosage.csv.gz"
        frame.to_csv(destination, index=False)
        print(f"{out_name}: {len(frame)} dong, {covered} mau -> {destination}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--methods", nargs="+",
                    default=["SNP2HLA", "CookHLA", "HIBAG", "DeepHLA", "AutoHLA"])
    # Hai co nay de arm pair song song voi arm nen trong CUNG thu muc predictions:
    # khong ghi de len nhau thi moi tinh duoc Delta (co pair - khong pair).
    ap.add_argument("--autohla-dir", default=None,
                    help="mac dinh results/ext_test/<o>/autohla/base")
    ap.add_argument("--autohla-name", default="AutoHLA",
                    help="ten file dosage ghi ra cho AutoHLA")
    args = ap.parse_args()
    build(args.config, args.methods, autohla_dir=args.autohla_dir,
          autohla_name=args.autohla_name)

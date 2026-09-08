#!/usr/bin/env python3
"""Canonical raw-chip metrics, following the DEEP*HLA paper.

Sensitivity/PPV use probabilistic allele dosage; F1 is derived from those
allele-level values; dosage r² uses Pearson correlation and Fisher-Z pooling;
concordance uses best-guess genotypes. Allele metrics are frequency-weighted.
"""
from collections import Counter
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from common import (LOCI, REPO, TARGET_COLS, code_to_2field, known_alleles,
                    load_label, normalize_allele)

# Bảng chính theo privateHLA; tên `maf_bin` được giữ trong CSV để tương thích mã cũ,
# nhưng đại lượng này là allele frequency (AF) riêng từng gene trong reference/train.
BINS = [(0, .01, "<1%"), (.01, .05, "1-5%"), (.05, .10, "5-10%"),
        (.10, .20, "10-20%"), (.20, 1.01, ">=20%")]
COOKHLA_BIN_NAMES = ["AF=0", "0-0.1%", "0.1-0.5%", "0.5-1%",
                     "1-5%", "5-10%", "10-20%", ">=20%"]
CHIPS = ["APMDA", "APMRA", "GSA"]
DATASETS = ["VN1K", "1KGP_VN1K"]
FOLDS = ["B301", "B303", "B304", "B305"]


def maf_bin(f):
    return next(name for lo, hi, name in BINS if lo <= f < hi)


def cookhla_af_bin(f):
    if f == 0: return "AF=0"
    if f < .001: return "0-0.1%"
    if f < .005: return "0.1-0.5%"
    if f < .01: return "0.5-1%"
    return maf_bin(f)


def ref_path(dataset, fold):
    root = "deephla_vn1k" if dataset == "VN1K" else "deephla_1kgp_vn1k"
    stem = "VN1K" if dataset == "VN1K" else "1KGP_VN1K"
    return Path(REPO, root, "ref", fold, f"{stem}.{fold}.bgl.phased.gz")


def train_freqs(dataset, fold):
    out = {}
    with gzip.open(ref_path(dataset, fold), "rt") as fh:
        for line in fh:
            fields = line.split()
            if fields[:1] != ["M"] or ":" not in fields[1] or not fields[1].startswith("HLA_"):
                continue
            gene, allele = fields[1][4:].split("_", 1)
            if gene in LOCI:
                out[(gene, normalize_allele(allele))] = fields[2:].count("P") / len(fields[2:])
    return out


def csv_dosages(path, label, dosage=None):
    imp = pd.read_csv(path, dtype=str, index_col=0).fillna("")
    shared = label.index.intersection(imp.index, sort=False)
    return dosage_rows(label.loc[shared], imp.reindex(shared), dosage)


def dosage_rows(truth, pred, dosage=None):
    rows = {}
    for gene in LOCI:
        universe = set()
        tc, pc = [], []
        for sid in truth.index:
            t = Counter(normalize_allele(truth.at[sid, f"{gene}_{i}"]) for i in (1, 2))
            p = Counter(normalize_allele(pred.at[sid, f"{gene}_{i}"]) for i in (1, 2))
            t.pop(None, None); p.pop(None, None)
            tc.append(t); pc.append(p); universe |= t.keys() | p.keys()
        for allele in universe:
            hard = np.array([x[allele] for x in pc], dtype=float)
            rows[(gene, allele)] = (
                np.array([x[allele] for x in tc], dtype=float),
                dosage.get((gene, allele), hard) if dosage else hard, hard)
    return rows


def vcf_dosages(path, label, cook=False):
    known = known_alleles("VN1K" if "VN1K" in str(path) and "1KGP" not in str(path)
                          else "1KGP_VN1K")
    samples, sums, counts = None, {}, Counter()
    paths = sorted(Path(path).parent.glob("*raw_imputation_out.vcf")) if cook else [Path(path)]
    for current in paths:
        with open(current) as fh:
            for line in fh:
                if line.startswith("#CHROM"):
                    samples = line.rstrip().split("\t")[9:]
                elif line.startswith("#"):
                    continue
                else:
                    f = line.rstrip().split("\t")
                    marker = f[2]
                    if cook:
                        parts = marker.split("_")
                        if len(parts) != 4 or not parts[3].startswith("exon"):
                            continue
                        gene, allele = parts[1], code_to_2field(parts[2], known.get(parts[1]))
                    else:
                        if not marker.startswith("HLA_") or ":" not in marker:
                            continue
                        gene, allele = marker[4:].split("_", 1)
                        allele = normalize_allele(allele)
                    if gene not in LOCI or not allele:
                        continue
                    fmt = f[8].split(":")
                    ds_i = fmt.index("DS")
                    values = np.array([2 - float(x.split(":")[ds_i]) for x in f[9:]])
                    key = (gene, allele)
                    sums[key] = sums.get(key, 0) + values
                    counts[key] += 1
    shared = label.index.intersection(samples, sort=False)
    return {(g, a): pd.Series(v, index=samples).reindex(shared).to_numpy() / counts[g, a]
            for (g, a), v in sums.items()}


def deephla_dosages(chip, dataset, fold, label):
    bundle = "deephla_vn1k" if dataset == "VN1K" else "deephla_1kgp_vn1k"
    stem = "VN1K" if dataset == "VN1K" else "1KGP_VN1K"
    base = Path(REPO, "results/imputed/raw_chip/deephla_sparse", chip, bundle, "out", fold)
    phased = pd.read_csv(base / f"{stem}.{fold}.deephla.phased",
                         sep="\t", header=None, index_col=0, dtype=str)
    phased.columns = range(phased.shape[1])
    fam = pd.read_csv(Path(REPO, bundle, "test", fold, "sample.fam"),
                      sep=r"\s+", header=None, dtype=str)
    ids = fam[1].tolist()
    pred = pd.DataFrame("", index=ids, columns=TARGET_COLS)
    for gene in LOCI:
        prefix = f"HLA_{gene}_"
        markers = [m for m in phased.index if m.startswith(prefix) and ":" in m]
        for sample_i, sid in enumerate(ids):
            for copy_i in (0, 1):
                hits = phased.loc[markers, 2 * sample_i + copy_i]
                hits = hits.index[hits.eq("P")]
                if len(hits):
                    pred.at[sid, f"{gene}_{copy_i + 1}"] = normalize_allele(hits[0][len(prefix):])
    shared = label.index.intersection(pred.index, sort=False)
    dose_file = base / f"{stem}.{fold}.deephla.dosage"
    dose = pd.read_csv(dose_file, sep="\t", header=None, index_col=0)
    dose = dose.iloc[:, 2:]
    dose.columns = ids
    probabilistic = {}
    for marker, values in dose.iterrows():
        if marker.startswith("HLA_") and ":" in marker:
            gene, allele = marker[4:].split("_", 1)
            if gene in LOCI:
                probabilistic[(gene, normalize_allele(allele))] = (
                    values.reindex(shared).astype(float).to_numpy())
    return dosage_rows(label.loc[shared], pred.loc[shared], probabilistic)


def hibag_dosages(chip, dataset, fold, label):
    base = Path(REPO, "results/imputed/raw_chip/HIBAG", chip, dataset, fold)
    shared = None
    pred = pd.DataFrame(index=[], columns=TARGET_COLS, dtype=str)
    for gene in LOCI:
        part = pd.read_csv(base / f"Imputed_test_{gene}.csv", dtype=str)
        part = part.set_index("SampleID")
        if shared is None:
            shared = part.index
            pred = pd.DataFrame(index=shared, columns=TARGET_COLS, dtype=str)
        else:
            shared = shared.intersection(part.index, sort=False)
        pred.loc[pred.index.intersection(part.index), f"{gene}_1"] = part["Pred_H1"]
        pred.loc[pred.index.intersection(part.index), f"{gene}_2"] = part["Pred_H2"]
    shared = label.index.intersection(pred.index, sort=False)
    return dosage_rows(label.loc[shared], pred.loc[shared])


def aehla_dosages(chip, dataset, fold, label, method_dir="AEHLA"):
    """ban goc: 4 group -> một bảng gọi cứng + dosage, do export_calls sinh.

    Mỗi group xuất riêng (g1=A, g2=B+C, g3=DPB1, g4=DRB1+DQA1+DQB1) nên phải ghép lại.
    Dosage là số bản sao kỳ vọng đã chuẩn hoá về tổng 2 trong mỗi khối gene — cùng ngữ
    nghĩa với trường DS mà các method khác cung cấp.
    """
    base = Path(REPO, "results/imputed/raw_chip", method_dir, chip, dataset, fold)
    pred, dosage, index = None, {}, None
    for g in (1, 2, 3, 4):
        calls_path, dose_path = base / f"g{g}.calls.csv", base / f"g{g}.dosage.csv"
        if not calls_path.exists():
            return {}
        calls = pd.read_csv(calls_path, index_col=0, dtype=str).fillna("")
        if pred is None:
            index = calls.index
            pred = pd.DataFrame("", index=index, columns=TARGET_COLS, dtype=object)
        for col in calls.columns:
            if col in TARGET_COLS:
                pred.loc[calls.index, col] = calls[col].map(normalize_allele)
        dose = pd.read_csv(dose_path, index_col=0)
        for marker, values in dose.iterrows():
            gene, allele = str(marker).split("_", 1)
            if gene in LOCI:
                dosage[(gene, normalize_allele(allele))] = values
    shared = label.index.intersection(pred.index, sort=False)
    dosage = {k: v.reindex(shared).astype(float).to_numpy() for k, v in dosage.items()}
    return dosage_rows(label.loc[shared], pred.loc[shared], dosage)


def summarize(rows, freqs, genes, bin_names=None, classifier=maf_bin, include_unseen=False):
    bin_names = bin_names or [name for _, _, name in BINS]
    selected = [(g, a, tv, dv, bv) for (g, a), (tv, dv, bv) in rows.items()
                if g in genes and (include_unseen or (g, a) in freqs)]
    stats = []
    for name in bin_names:
        alleles = [(a, tv, dv, bv) for g, a, tv, dv, bv in selected
                   if classifier(freqs.get((g, a), 0.0)) == name]
        if not alleles:
            continue
        per_allele = []
        for _, t, d, b in alleles:
            m = t.sum()
            soft_tp = np.minimum(t, d).sum()
            hard_tp = np.minimum(t, b).sum()
            r = np.corrcoef(t, d)[0, 1] if t.std() and d.std() else np.nan
            per_allele.append((m, soft_tp, d.sum(), hard_tp, r))
        true = sum(x[0] for x in per_allele)
        soft_tp = sum(x[1] for x in per_allele)
        predicted = sum(x[2] for x in per_allele)
        hard_tp = sum(x[3] for x in per_allele)
        se_num = sum(m * (tp / m) for m, tp, _, _, _ in per_allele if m)
        ppv_num = sum(m * (tp / pred) for m, tp, pred, _, _ in per_allele if m and pred)
        f1_num = sum(m * (2 * tp / (m + pred)) for m, tp, pred, _, _ in per_allele
                     if m and m + pred)
        z = [(np.arctanh(np.clip(r, -.999999, .999999)), m)
             for m, _, _, _, r in per_allele if m and not np.isnan(r)]
        safe = lambda n, d: n / d if d else np.nan
        stats.append(dict(maf_bin=name, n_alleles=len(alleles),
                          true_copies=int(true), predicted_copies=predicted,
                          tp=soft_tp, sn=safe(se_num, true), ppv=safe(ppv_num, true),
                          f1=safe(f1_num, true),
                          r2=np.tanh(safe(sum(v * w for v, w in z),
                                          sum(w for _, w in z))) ** 2,
                          concordance=safe(hard_tp, true),
                          sn_num=se_num, ppv_num=ppv_num, f1_num=f1_num,
                          cr_num=hard_tp, r_z_num=sum(v * w for v, w in z),
                          r_weight=sum(w for _, w in z)))
    return stats


def aggregate(result):
    overall = result[result.locus.eq("OVERALL")]
    summary = overall.groupby(["dataset", "method", "maf_bin"], sort=False)[
        ["n_alleles", "true_copies", "predicted_copies", "tp", "sn_num", "ppv_num",
         "f1_num", "cr_num", "r_z_num", "r_weight"]].sum().reset_index()
    summary["sn"] = summary.sn_num / summary.true_copies
    summary["ppv"] = summary.ppv_num / summary.true_copies
    summary["f1"] = summary.f1_num / summary.true_copies
    summary["r2"] = np.tanh(summary.r_z_num / summary.r_weight) ** 2
    summary["concordance"] = summary.cr_num / summary.true_copies
    return summary.drop(columns=["sn_num", "ppv_num", "f1_num", "cr_num",
                                 "r_z_num", "r_weight"])


def write_reports(dest, summary, detailed):
    cols = ["dataset", "method", "maf_bin", "true_copies", "sn", "ppv", "f1",
            "r2", "concordance"]
    (dest / "BAO_CAO_TONG_HOP_AF.md").write_text(
        "# Benchmark raw-chip theo allele-frequency bin\n\n"
        "AF tính riêng từng gene từ reference của chính ô; cùng AF được dùng cho mọi "
        "method. Bảng chính theo privateHLA: `<1%`, `1-5%`, `5-10%`, `10-20%`, "
        "`>=20%`. Metrics gộp micro/weighted theo giao thức chung.\n\n```\n"
        + summary[cols].round(5).to_string(index=False) + "\n```\n"
    )
    (dest / "BAO_CAO_CHI_TIET_AF.md").write_text(
        "# Rare-AF chi tiết theo CookHLA\n\n"
        "`AF=0` là allele có trong truth/prediction nhưng vắng khỏi reference của ô; "
        "bin này được tách khỏi allele hiếm có hỗ trợ trong reference. Bảng per-cell × "
        "locus đầy đủ nằm trong `metrics_by_af_bin_cookhla.csv`.\n\n```\n"
        + detailed[cols].round(5).to_string(index=False) + "\n```\n"
    )


def main():
    out, detailed = [], []
    for dataset in DATASETS:
        label = load_label(dataset)
        for fold in FOLDS:
            freqs = train_freqs(dataset, fold)
            for chip in CHIPS:
                paths = {
                    "SNP2HLA": Path(REPO, "results/imputed/raw_chip/SNP2HLA", chip,
                                    dataset, fold, "test/converted.csv"),
                    "CookHLA": Path(REPO, "results/imputed/raw_chip/CookHLA", chip,
                                   dataset, fold, "test/converted.csv")}
                method_rows = {m: csv_dosages(p, label) for m, p in paths.items()}
                snp_vcf = Path(REPO, "methods/SNP2HLA_UM/SNP2HLA_UM/example_input/hg38/"
                               f"snp_runs/rawchip_20260727d.{chip}.{dataset}.{fold}.test/"
                               f"rawchip_20260727d.{chip}.{dataset}.{fold}.test.bgl.vcf")
                cook_vcf = Path(REPO, "methods/CookHLA/_cookhla_bundle/raw_chip",
                                chip, dataset, fold, "test/x.raw_imputation_out.vcf")
                method_rows["SNP2HLA"] = csv_dosages(
                    paths["SNP2HLA"], label, vcf_dosages(snp_vcf, label))
                method_rows["CookHLA"] = csv_dosages(
                    paths["CookHLA"], label, vcf_dosages(cook_vcf, label, cook=True))
                method_rows["DeepHLA"] = deephla_dosages(chip, dataset, fold, label)
                method_rows["HIBAG"] = hibag_dosages(chip, dataset, fold, label)
                # AEHLA_PAIR_WARM = Việc 0: đúng luật chọn checkpoint của
                # `pair_full_warm` trên lưới 10-fold, xem run_pair_energy_warm_24cells.sh
                for method, method_dir in (("AEHLA", "AEHLA"),
                                           ("AEHLA_PAIR", "AEHLA_PAIR"),
                                           ("AEHLA_PAIR_WARM", "AEHLA_PAIR_WARM")):
                    decoded = aehla_dosages(chip, dataset, fold, label, method_dir)
                    if decoded:       # bỏ qua ô chưa chạy
                        method_rows[method] = decoded
                for method, rows in method_rows.items():
                    for locus in LOCI + ["OVERALL"]:
                        genes = LOCI if locus == "OVERALL" else [locus]
                        for stat in summarize(rows, freqs, genes):
                            out.append(dict(dataset=dataset, chip=chip, fold=fold,
                                            method=method, locus=locus, **stat))
                        for stat in summarize(rows, freqs, genes, COOKHLA_BIN_NAMES,
                                              cookhla_af_bin, include_unseen=True):
                            detailed.append(dict(dataset=dataset, chip=chip, fold=fold,
                                                 method=method, locus=locus, **stat))
    result = pd.DataFrame(out)
    detail_result = pd.DataFrame(detailed)
    dest = Path(REPO, "results/metrics/raw_chip_maf")
    dest.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest / "metrics_by_maf_bin.csv", index=False)
    result.to_csv(dest / "metrics_by_af_bin.csv", index=False)
    detail_result.to_csv(dest / "metrics_by_af_bin_cookhla.csv", index=False)
    summary = aggregate(result)
    summary.to_csv(dest / "summary_by_maf_bin.csv", index=False)
    summary.to_csv(dest / "summary_by_af_bin.csv", index=False)
    detail_summary = aggregate(detail_result)
    detail_summary.to_csv(dest / "summary_by_af_bin_cookhla.csv", index=False)
    write_reports(dest, summary, detail_summary)
    print(f"primary={len(result)}, CookHLA-detail={len(detail_result)} rows -> {dest}")


if __name__ == "__main__":
    assert [maf_bin(x) for x in (0, .01, .05, .10, .20)] == [b[2] for b in BINS]
    assert [cookhla_af_bin(x) for x in (0, .000999999, .001, .004999999, .005, .009999999, .01, .20)] == [
        "AF=0", "0-0.1%", "0.1-0.5%", "0.1-0.5%", "0.5-1%", "0.5-1%", "1-5%", ">=20%"]
    main()

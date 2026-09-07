#!/usr/bin/env python3
"""Bộ metrics CHUẨN HOÁ cho MỌI phương pháp trên giao thức 10-fold CV VN1K/APMDA.

Một allele universe, một bảng AF, một evaluator cho tất cả:

- universe = mọi allele 2-field xuất hiện trong nhãn cohort VN1K của từng gene;
  phương pháp nào không phát ra allele đó thì dosage = 0 (không bị loại khỏi mẫu số).
  Đây là chỗ bảng cũ để HIBAG thiếu 54 bản sao ở bin `<1%`.
- AF = số bản sao allele / tổng bản sao hợp lệ CỦA CHÍNH GENE, tính từ train của
  từng fold (CLAUDE.md 2026-08-18). Validation/test không tham gia.
- micro theo AF bin, TP = min(truth, dosage); `sn` micro, `ppv`/`f1` trung bình
  theo allele có trọng số = số bản sao thật; `concordance` trên hard call.

    python scripts/eval/cv10_canonical_metrics.py pool     # gộp AEHLA_PAIR
    python scripts/eval/cv10_canonical_metrics.py score
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/eval"), str(ROOT / "scripts/cv_vn1k"),
                str(ROOT / "AEHLA")]
import common  # noqa: E402
import compute_raw_chip_maf_metrics as metric  # noqa: E402

# Cohort/thu muc CV mo ra bang env de HAN cham qua DUNG evaluator nay -- mot
# allele universe, mot bang AF, mot luat goi cung cho moi method. Mac dinh giu
# nguyen VN1K nen moi bang cu tai lap khong doi.
COHORT = os.environ.get("CV10_COHORT", "VN1K")
CV = ROOT / os.environ.get("CV10_CV_DIR", "results/cv_vn1k")
PRED = ROOT / os.environ.get("CV10_PRED_DIR", str((CV / "predictions").relative_to(ROOT)))
OUT = ROOT / os.environ.get("CV10_OUT", "results/cv10_canonical")
FOLDS = [int(x) for x in os.environ.get("CV10_FOLDS", "1,2,3,4,5,6,7,8,9,10").split(",")]
# tên hiển thị -> file gộp trong results/cv_vn1k/predictions
METHODS = {
    "SNP2HLA": "SNP2HLA",
    "CookHLA": "CookHLA",
    "DEEP*HLA": "DeepHLA",
    "HIBAG": "HIBAG",
    "AEHLA": "base",
    "AEHLA+pair": "AEHLA_PAIR",
}
# CV10_METHODS THAY HAN danh sach (cohort khac co bo baseline khac -- HAN khong
# co file SNP2HLA/CookHLA trong thu muc du doan cua no). CV10_EXTRA_METHODS thi
# BO SUNG vao danh sach dang co.
_replace = dict(item.split("=") for item in
                os.environ.get("CV10_METHODS", "").split(",") if item)
if _replace:
    METHODS = _replace
METHODS.update(dict(item.split("=") for item in
                    os.environ.get("CV10_EXTRA_METHODS", "").split(",") if item))
# CV10_LOCI thu hep danh sach gene duoc CHAM. Mac dinh = ca 7 gene, nen moi bang
# cu tai lap khong doi. Dung khi mot locus co nhan hong o cohort do: DQA1 cua HAN
# co F = 0.413 (hom 0.492 vs HWE 0.133) trong khi 6 gene kia F <= 0.066 -- cham no
# la cham xem method tai lap loi cua bo goi gioi den dau.
LOCI = [g for g in os.environ.get("CV10_LOCI", ",".join(common.LOCI)).split(",") if g]
assert set(LOCI) <= set(common.LOCI), f"CV10_LOCI khong hop le: {LOCI}"
BIN_ORDER = [name for _, _, name in metric.BINS]
SUM_COLUMNS = ["n_alleles", "true_copies", "predicted_copies", "tp",
               "sn_num", "ppv_num", "f1_num", "cr_num", "r_z_num", "r_weight"]
PRED_COLUMNS = ["sample_id", "fold", "gene", "allele", "dosage"]


# ---------------------------------------------------------------------- pool ---
def pool_pair(source=ROOT / "results/pair_energy_causal/predictions",
              arm="pair_full", name="AEHLA_PAIR", seed=77):
    """pair_full_warm: dosage theo chỉ số allele -> định dạng cv_pool chung."""
    from objects.encoder import Encoder

    models = Path("results/latent_soft_knn_cv/models")
    rows = []
    for fold in FOLDS:
        for group in (1, 2, 3, 4):
            path = Path(source) / f"fold{fold:02d}_group{group}_{arm}.csv.gz"
            if not path.exists():
                raise SystemExit(f"thiếu {path}")
            enc = Encoder()
            enc.load(ROOT / models / f"fold{fold:02d}/s{seed}/g{group}"
                     / f"models_4digit/collapse/group_{group}/encoder.pkl")
            frame = pd.read_csv(path)
            samples = frame.pop("sample_id").to_numpy()
            for column in frame.columns:
                head, index = column.rsplit(":", 1)
                gene = head.replace("HLA_", "")
                allele = enc.decoder[(gene, int(index))]
                values = frame[column].to_numpy(dtype=float)
                keep = values > 1e-3
                rows.extend(zip(samples[keep], [fold] * int(keep.sum()),
                                [gene] * int(keep.sum()), [allele] * int(keep.sum()),
                                values[keep].round(4)))
    frame = pd.DataFrame(rows, columns=["sample_id", "fold", "gene", "allele", "dosage"])
    assert frame.groupby("sample_id").fold.nunique().eq(1).all(), "mẫu nằm ở >1 fold"
    destination = PRED / f"{name}.dosage.csv.gz"
    frame.to_csv(destination, index=False)
    print(f"{name}: {len(frame)} dòng, {frame.sample_id.nunique()} mẫu -> {destination}")


# --------------------------------------------------------------------- score ---
def fold_split(fold):
    protocol = CV / "protocol" / f"fold{fold:02d}"
    return (protocol / "train.samples").read_text().split(), \
           (protocol / "test.samples").read_text().split()


def universe(labels):
    """{gene: [allele, ...]} lấy từ nhãn cohort — dùng chung cho mọi phương pháp."""
    out = {}
    for gene in LOCI:
        values = labels[[f"{gene}_1", f"{gene}_2"]].to_numpy().ravel()
        out[gene] = sorted({v for v in values if v})
    return out


def fold_freqs(labels, train_ids, alleles):
    """AF theo gene, tính riêng trên train của fold; allele không thấy -> 0.0."""
    freqs = {}
    for gene, names in alleles.items():
        values = pd.Series(labels.loc[train_ids, [f"{gene}_1", f"{gene}_2"]]
                           .to_numpy().ravel())
        values = values[values.notna() & values.ne("")]
        counts = values.value_counts()
        total = len(values)
        for allele in names:
            freqs[(gene, allele)] = counts.get(allele, 0) / total if total else 0.0
    return freqs


def best_guess(dosage, gene, names, n):
    """Hard call = hai bản sao có dosage dư lớn nhất (cho phép đồng hợp)."""
    matrix = np.column_stack([dosage[(gene, a)] for a in names])
    work = matrix.copy()
    picks = []
    for _ in range(2):
        hit = work.argmax(1)
        picks.append(hit)
        work[np.arange(n), hit] -= 1
    return [[names[i] for i in pick] for pick in picks]


def method_rows(pooled, fold, labels, alleles, test_ids):
    part = pooled[pooled.fold.eq(fold)]
    dosage, pred = {}, pd.DataFrame("", index=test_ids, columns=common.TARGET_COLS)
    for gene, names in alleles.items():
        block = part[part.gene.eq(gene)]
        wide = (block.pivot_table(index="sample_id", columns="allele", values="dosage",
                                  aggfunc="sum")
                .reindex(index=test_ids, columns=names).fillna(0.0))
        for allele in names:
            dosage[(gene, allele)] = wide[allele].to_numpy(dtype=float)
        first, second = best_guess(dosage, gene, names, len(test_ids))
        pred[f"{gene}_1"], pred[f"{gene}_2"] = first, second
    rows = metric.dosage_rows(labels.loc[test_ids], pred, dosage)
    # Mỗi arm phải được chấm trên đúng cùng universe, kể cả allele không có truth,
    # hard-call hay dosage dương trong test fold này.
    for gene, names in alleles.items():
        for allele in names:
            rows.setdefault((gene, allele),
                            (np.zeros(len(test_ids)), dosage[(gene, allele)],
                             np.zeros(len(test_ids))))
    return rows


def collapse_labels(labels):
    """Gop nhan ve ten dai dien G-group (HLA_GGROUP=1); tat thi tra nguyen."""
    if not common.GGROUP:
        return labels
    gmap = common.ggroup_map()
    for gene in common.LOCI:
        for i in (1, 2):
            column = f"{gene}_{i}"
            labels[column] = [gmap.get((gene, a), a) if a else a
                              for a in labels[column]]
    return labels


def validate_pooled(name, frame, labels):
    missing = set(PRED_COLUMNS) - set(frame.columns)
    if missing:
        raise AssertionError(f"{name}: thiếu cột {sorted(missing)}")
    frame = frame[PRED_COLUMNS].copy()
    # ID mau HAN la so nen pandas doc thanh int64, con protocol/nhan tra chuoi --
    # hai tap khong bao gio bang nhau va assert coverage duoi day bao "khong khop"
    # trong khi du lieu hoan toan dung. Cung bay ma blend.py da phai ne.
    frame["sample_id"] = frame.sample_id.astype(str)
    # Nhan HAN ghi "A*33:03" va common.load_label boc tien to truoc khi dung
    # universe; file du doan thi giu nguyen. Ap CUNG luat boc o day, khong thi moi
    # allele HAN deu bi bao "ngoai universe" -- cung lech ma cong C3/C4 da gap.
    frame["allele"] = (frame.allele
                       .map(lambda v: v.split("*", 1)[1]
                            if isinstance(v, str) and "*" in v else v)
                       .map(common.normalize_allele))
    # Gop TRUOC groupby: hai ten cua cung mot G-group cong dosage lai voi nhau,
    # roi hard-call moi chay -- neu gop sau thi method nao chia lieu cho ca hai
    # ten se bi phat, ma cai lech do la ten chu khong phai sinh hoc.
    if common.GGROUP:
        gmap = common.ggroup_map()
        frame["allele"] = [gmap.get((g, a), a)
                           for g, a in zip(frame.gene, frame.allele)]
    if frame.allele.isna().any() or not np.isfinite(frame.dosage).all():
        raise AssertionError(f"{name}: allele/dosage không hợp lệ")
    frame = frame.groupby(["sample_id", "fold", "gene", "allele"], as_index=False).dosage.sum()
    if frame.dosage.lt(0).any() or frame.dosage.gt(2.0001).any():
        raise AssertionError(f"{name}: dosage ngoài [0,2]")

    fold_of = {sid: fold for fold in FOLDS for sid in fold_split(fold)[1]}
    expected = set(fold_of)
    if set(frame.sample_id) != expected:
        raise AssertionError(f"{name}: coverage mẫu không khớp test fold")
    bad_fold = frame.fold.ne(frame.sample_id.map(fold_of))
    if bad_fold.any():
        raise AssertionError(f"{name}: có prediction gắn sai fold")
    frame = frame[frame.gene.isin(LOCI)]
    if set(frame.gene) != set(LOCI):
        raise AssertionError(f"{name}: gene không đủ/không hợp lệ")
    observed = frame[["sample_id", "gene"]].drop_duplicates()
    if len(observed) != len(expected) * len(LOCI):
        raise AssertionError(f"{name}: thiếu sample-gene")

    alleles = universe(labels)
    alien = {(g, a) for g, a in zip(frame.gene, frame.allele) if a not in alleles[g]}
    if alien:
        raise AssertionError(f"{name}: allele ngoài universe: {sorted(alien)[:5]}")
    return frame


def score():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = collapse_labels(common.load_label(COHORT).map(common.normalize_allele))
    alleles = universe(labels)
    pooled = {name: validate_pooled(name, pd.read_csv(PRED / f"{stem}.dosage.csv.gz"),
                                    labels)
              for name, stem in METHODS.items()}

    standard, rare = [], []
    for fold in FOLDS:
        train_ids, test_ids = fold_split(fold)
        freqs = fold_freqs(labels, train_ids, alleles)
        for name in METHODS:
            rows = method_rows(pooled[name], fold, labels, alleles, test_ids)
            for gene in LOCI:
                standard.extend(
                    dict(fold=fold, gene=gene, arm=name, **stat)
                    for stat in metric.summarize(rows, freqs, [gene]))
                rare.extend(
                    dict(fold=fold, gene=gene, arm=name, **stat)
                    for stat in metric.summarize(
                        rows, freqs, [gene], metric.COOKHLA_BIN_NAMES,
                        metric.cookhla_af_bin, include_unseen=True))
        print(f"xong fold {fold:02d}", flush=True)
    audit = pd.DataFrame([
        dict(arm=name, rows=len(frame), samples=frame.sample_id.nunique(),
             folds=frame.fold.nunique(), genes=frame.gene.nunique(),
             dosage_min=frame.dosage.min(), dosage_max=frame.dosage.max())
        for name, frame in pooled.items()
    ])
    return pd.DataFrame(standard), pd.DataFrame(rare), audit


def reduce_metrics(raw, dimensions):
    out = raw.groupby(dimensions + ["arm", "maf_bin"], sort=False)[
        SUM_COLUMNS].sum().reset_index()
    out["sn"] = out.sn_num / out.true_copies
    out["ppv"] = out.ppv_num / out.true_copies
    out["f1"] = out.f1_num / out.true_copies
    out["concordance"] = out.cr_num / out.true_copies
    ratio = np.divide(out.r_z_num, out.r_weight, out=np.full(len(out), np.nan),
                      where=out.r_weight > 0)
    out["r2"] = np.tanh(ratio) ** 2
    return out


def paired(fold_rows, order, unit="fold"):
    rows = []
    arms = list(METHODS)
    for candidate in arms:
        for control in arms:
            if candidate == control:
                continue
            for name in order:
                wide = fold_rows[fold_rows.maf_bin.eq(name)].pivot(
                    index=unit, columns="arm", values="f1")
                if candidate not in wide or control not in wide:
                    continue
                pair = wide[[candidate, control]].dropna()
                delta = pair[candidate] - pair[control]
                p = (1.0 if len(delta) < 2 or np.allclose(delta, 0)
                     else float(wilcoxon(pair[candidate], pair[control]).pvalue))
                rows.append(dict(candidate=candidate, control=control, maf_bin=name,
                                 n_folds=len(delta), wins=int((delta > 0).sum()),
                                 mean_delta_f1=float(delta.mean()),
                                 median_delta_f1=float(delta.median()),
                                 wilcoxon_p=p))
    return pd.DataFrame(rows)


def check_common_denominator(fold_rows, order):
    """Mọi phương pháp phải có cùng universe và true_copies trong từng fold/bin."""
    for column in ("n_alleles", "true_copies"):
        spread = fold_rows[fold_rows.maf_bin.isin(order)].groupby(
            ["fold", "maf_bin"])[column].nunique()
        bad = spread[spread > 1]
        if len(bad):
            raise AssertionError(f"{column} lệch giữa các arm:\n{bad}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("pool", "score"))
    args = parser.parse_args()
    if args.step == "pool":
        pool_pair()
        return
    standard_gene, rare_gene, audit = score()
    standard_fold = reduce_metrics(standard_gene, ["fold"])
    rare_fold = reduce_metrics(rare_gene, ["fold"])
    check_common_denominator(standard_fold, BIN_ORDER)
    check_common_denominator(rare_fold, metric.COOKHLA_BIN_NAMES)
    summary = reduce_metrics(standard_gene, [])
    rare_summary = reduce_metrics(rare_gene, [])
    summary = summary.merge(
        standard_fold.groupby(["arm", "maf_bin"]).f1.std()
        .rename("f1_fold_sd").reset_index(), on=["arm", "maf_bin"], how="left")
    rare_summary = rare_summary.merge(
        rare_fold.groupby(["arm", "maf_bin"]).f1.std()
        .rename("f1_fold_sd").reset_index(), on=["arm", "maf_bin"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    standard_gene.to_csv(OUT / "metrics_by_locus_fold_af_bin.csv", index=False)
    rare_gene.to_csv(OUT / "metrics_by_locus_fold_rare_af_bin.csv", index=False)
    standard_fold.to_csv(OUT / "metrics_by_fold_af_bin.csv", index=False)
    rare_fold.to_csv(OUT / "metrics_by_fold_rare_af_bin.csv", index=False)
    summary.to_csv(OUT / "summary_by_af_bin.csv", index=False)
    rare_summary.to_csv(OUT / "summary_by_rare_af_bin.csv", index=False)
    audit.to_csv(OUT / "input_audit.csv", index=False)
    # Mot fold (external test) thi khong con don vi lap nao theo fold; lay
    # LOCUS lam don vi ghep doi de van co kiem dinh, va ghi ro trong bao cao.
    unit = "fold" if len(FOLDS) > 1 else "gene"
    std_unit = standard_fold if unit == "fold" else reduce_metrics(standard_gene, ["gene"])
    rare_unit = rare_fold if unit == "fold" else reduce_metrics(rare_gene, ["gene"])
    paired(std_unit, BIN_ORDER, unit).to_csv(OUT / "paired_wilcoxon.csv", index=False)
    paired(rare_unit, metric.COOKHLA_BIN_NAMES, unit).to_csv(
        OUT / "paired_wilcoxon_rare.csv", index=False)
    cols = ["arm", "maf_bin", "n_alleles", "true_copies", "sn", "ppv", "f1",
            "concordance", "f1_fold_sd"]
    print(summary[cols].round(5).to_string(index=False))


if __name__ == "__main__":
    main()

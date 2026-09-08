#!/usr/bin/env python3
"""Giao thức chấm điểm chung cho MỌI phương pháp chạy CV VN1K/APMDA 10 fold.

Hai bước tách rời:

1. `pool`  — gộp dự đoán test của 10 fold thành MỘT file duy nhất
             `do noi bo cohort<method>.dosage.csv.gz`.
             Mỗi mẫu trong cohort xuất hiện đúng một lần (là test của đúng một fold),
             nên file gộp là một dự đoán hoàn chỉnh trên toàn bộ 996 mẫu.

2. `score` — chấm trên file gộp: micro theo từng MAF bin, `TP = min(truth, dosage)`.
             Vì gộp rồi thì mỗi allele phải nằm ở đúng một bin, tần suất chia bin lấy
             từ NHÃN của toàn cohort VN1K (không phải train của từng fold — mỗi fold
             một tần suất khác nhau thì không gộp được). Train mỗi fold = 85% cohort
             nên hai cách chỉ lệch ở vài allele sát ranh giới bin.

Định dạng file gộp (phương pháp mới chỉ cần xuất đúng 4 cột này là chấm được):

    sample_id,fold,gene,allele,dosage

- một dòng cho mỗi (mẫu, allele) có dosage > 1e-3; thiếu dòng nghĩa là dosage 0
- `gene` thuộc A/B/C/DPB1/DRB1/DQA1/DQB1, `allele` là 2 field đã chuẩn hoá ("11:01")
- `dosage` là số bản sao kỳ vọng trong [0, 2]; hard call thì dùng 0/1/2

Dùng:
    python cv_pool pool --method SNP2HLA CookHLA
    python cv_pool score
"""
import argparse
import os
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from common import LOCI, code_to_2field, known_alleles, normalize_allele  # noqa: E402

# Bảng bin: `legacy` giữ nguyên số đã công bố; `standard`/`rare` là chuẩn AF bin
# 2026-08-18 (CLAUDE.md). Chọn bằng CV_POOL_BIN_SCHEME; scheme khác legacy ghi ra file
# có hậu tố riêng nên không đè kết quả cũ.
BIN_SCHEMES = {
    "legacy": ((0, .01, "<1%"), (.01, .02, "1-2%"), (.02, .05, "2-5%"),
               (.05, .10, "5-10%"), (.10, 1.01, ">=10%")),
    "standard": ((0, .01, "<1%"), (.01, .05, "1-5%"), (.05, .10, "5-10%"),
                 (.10, .20, "10-20%"), (.20, 1.01, ">=20%")),
    "rare": ((0, 0, "AF=0"), (0, .001, "0-0.1%"), (.001, .005, "0.1-0.5%"),
             (.005, .01, "0.5-1%"), (.01, .05, "1-5%"), (.05, .10, "5-10%"),
             (.10, .20, "10-20%"), (.20, 1.01, ">=20%")),
}
SCHEME = os.environ.get("CV_POOL_BIN_SCHEME", "legacy")
if SCHEME not in BIN_SCHEMES:
    raise SystemExit(f"CV_POOL_BIN_SCHEME phải thuộc {sorted(BIN_SCHEMES)}")
BINS = BIN_SCHEMES[SCHEME]
SUFFIX = "" if SCHEME == "legacy" else f"_{SCHEME}"


def bin_name(freq):
    """AF=0 là bin riêng (allele chưa từng thấy), nên không dùng được lo <= f < hi."""
    for lo, hi, name in BINS:
        if lo == hi == 0:
            if freq == 0:
                return name
        elif name == "0-0.1%":
            if 0 < freq < hi:
                return name
        elif lo <= freq < hi:
            return name
    raise ValueError(f"AF {freq} không rơi vào bin nào của scheme {SCHEME}")
BASE = ROOT / "results/cv_vn1k"
PRED = BASE / "predictions"
OUT = BASE / "metrics"
LABEL = ROOT / "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv"
COLUMNS = ["sample_id", "fold", "gene", "allele", "dosage"]
# Cùng bảng màu với `plot_raw_chip_maf_metrics` để hai hình đọc chung được.
COLORS = {"SNP2HLA": "#0072B2", "CookHLA": "#009E73", "DeepHLA": "#D55E00",
          "HIBAG": "#7B2CBF", "AEHLA": "#0b0b0b",
          "base": "#0b0b0b", "haprec": "#CC79A7", "win500": "#E69F00"}
MIN_DOSAGE = 1e-3
FOLDS = range(1, 11)


# --------------------------------------------------------------------- pool ---
def vcf_dosages(paths, cook, known=None):
    """DS trong VCF của SNP2HLA/CookHLA -> {(gene, allele): Series theo sample}."""
    known = known or known_alleles("VN1K")
    samples, sums, counts = None, {}, Counter()
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if line.startswith("#CHROM"):
                    samples = line.rstrip().split("\t")[9:]
                    continue
                if line.startswith("#"):
                    continue
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
                if "DS" not in fmt:
                    continue
                ds = fmt.index("DS")
                # A1 của panel là allele "vắng mặt" -> số bản sao = 2 - DS.
                values = np.array([2 - float(x.split(":")[ds]) for x in f[9:]])
                key = (gene, allele)
                sums[key] = sums.get(key, 0) + values
                counts[key] += 1
    # CookHLA cho mỗi allele nhiều marker exon -> lấy trung bình, giống giao thức 24 ô.
    return {k: pd.Series(v / counts[k], index=samples) for k, v in sums.items()}


def fold_paths(method, fold):
    d = BASE / "baselines" / f"fold{fold:02d}"
    if method == "SNP2HLA":
        return [d / "snp2hla" / f"fold{fold:02d}.bgl.vcf"], False
    if method == "CookHLA":
        return sorted((d / "cookhla").glob("*raw_imputation_out.vcf")), True
    raise SystemExit(f"chưa biết cách đọc output của {method}")


def pool(method):
    rows = []
    for fold in FOLDS:
        if method == "DeepHLA":
            # DEEP*HLA VN1K adapter: dosage rows already contain soft copies.
            d = BASE / "deephla" / f"fold{fold:02d}"
            dosef = d / f"VN1K.fold{fold:02d}.deephla.dosage"
            fam = pd.read_csv(d / "test.fam", sep=r"\s+", header=None, dtype=str)
            ids = fam[1].tolist(); raw = pd.read_csv(dosef, sep="\t", header=None, index_col=0)
            vals = raw.iloc[:, 2:]; vals.columns = ids
            rows2=[]
            known = known_alleles("VN1K")
            for marker, row in vals.iterrows():
                if not str(marker).startswith("HLA_") or ":" not in str(marker): continue
                gene, code = str(marker)[4:].split("_", 1)
                if gene not in LOCI: continue
                allele = normalize_allele(code_to_2field(code, known.get(gene)))
                for sid, value in row.items():
                    if float(value) > MIN_DOSAGE: rows2.append((sid, fold, gene, allele, round(float(value),4)))
            rows.extend(rows2); continue
        if method == "HIBAG":
            test_ids = (BASE / "protocol" / f"fold{fold:02d}" / "test.samples").read_text().split()
            for gene in LOCI:
                f = BASE / "hibag" / f"fold{fold:02d}" / f"Imputed_test_{gene}.csv"
                q = pd.read_csv(f, dtype=str).set_index("SampleID")
                for sid, row in q.iterrows():
                    for col in ("Pred_H1", "Pred_H2"):
                        a = normalize_allele(row[col])
                        if a: rows.append((sid, fold, gene, a, 1.0))
            continue
        paths, cook = fold_paths(method, fold)
        paths = [p for p in paths if p.exists()]
        test_ids = (BASE / "protocol" / f"fold{fold:02d}" / "test.samples").read_text().split()
        if not paths:
            raise SystemExit(f"{method} fold{fold:02d}: chưa có output")
        dose = vcf_dosages(paths, cook)
        for (gene, allele), series in dose.items():
            values = series.reindex(test_ids)
            keep = values.notna() & (values > MIN_DOSAGE)
            for sid, value in values[keep].items():
                rows.append((sid, fold, gene, allele, round(float(value), 4)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    covered = df.sample_id.nunique()
    expected = sum(len((BASE / "protocol" / f"fold{f:02d}" / "test.samples").read_text().split())
                   for f in FOLDS)
    assert df.groupby("sample_id").fold.nunique().eq(1).all(), "một mẫu nằm ở >1 fold"
    PRED.mkdir(parents=True, exist_ok=True)
    dest = PRED / f"{method}.dosage.csv.gz"
    df.to_csv(dest, index=False)
    print(f"{method}: {len(df)} dòng, {covered}/{expected} mẫu -> {dest}")


# -------------------------------------------------------------------- score ---
def truth_matrix(labels, ids):
    out = {}
    for gene in LOCI:
        for i, sid in enumerate(ids):
            for copy in (1, 2):
                allele = normalize_allele(labels.at[sid, f"{gene}_{copy}"])
                if allele:
                    out.setdefault((gene, allele), np.zeros(len(ids)))[i] += 1
    return out


def micro(pred, truth, mask):
    p, t = pred[:, mask], truth[:, mask]
    called, true, tp = p.sum(), t.sum(), np.minimum(p, t).sum()
    # R² liều: Pearson theo từng allele rồi gộp Fisher-Z có trọng số = số bản sao thật,
    # đúng công thức của `compute_raw_chip_maf_metrics.py` để so được với lưới 24 ô.
    z = []
    for j in range(t.shape[1]):
        m = t[:, j].sum()
        if m and t[:, j].std() and p[:, j].std():
            r = np.corrcoef(t[:, j], p[:, j])[0, 1]
            z.append((np.arctanh(np.clip(r, -.999999, .999999)), m))
    r2 = np.tanh(sum(v * w for v, w in z) / sum(w for _, w in z)) ** 2 if z else np.nan
    return dict(true_copies=float(true), predicted_copies=float(called), tp=float(tp),
                sn=tp / true if true else np.nan,
                ppv=tp / called if called else np.nan,
                f1=2 * tp / (true + called) if true + called else np.nan,
                r2=r2)


def score(methods):
    labels = pd.read_csv(LABEL, dtype=str).set_index("Sample ID")
    fold_of = {}
    for fold in FOLDS:
        for sid in (BASE / "protocol" / f"fold{fold:02d}" / "test.samples").read_text().split():
            fold_of[sid] = fold
    ids = sorted(fold_of)
    truth_map = truth_matrix(labels, ids)
    freq = {k: v.sum() / (2 * len(ids)) for k, v in truth_map.items()}

    preds = {}
    for method in methods:
        path = PRED / f"{method}.dosage.csv.gz"
        if not path.exists():
            print(f"bỏ qua {method}: chưa có {path.name}")
            continue
        preds[method] = pd.read_csv(path)
    if not preds:
        raise SystemExit("không có file gộp nào để chấm")

    # Vũ trụ allele = allele có trong nhãn + allele bất kỳ method nào gọi ra (freq 0 ->
    # bin <1%), để dương tính giả trên allele lạ vẫn bị tính vào predicted_copies.
    keys = set(truth_map) | {(g, a) for df in preds.values()
                             for g, a in zip(df.gene, df.allele)}
    keys = sorted(keys)
    index = {k: i for i, k in enumerate(keys)}
    position = {sid: i for i, sid in enumerate(ids)}
    truth = np.zeros((len(ids), len(keys)))
    for k, v in truth_map.items():
        truth[:, index[k]] = v
    bins = np.array([bin_name(freq.get(k, 0.)) for k in keys])

    rows, fold_rows, locus_rows = [], [], []
    fold_vec = np.array([fold_of[sid] for sid in ids])
    gene_vec = np.array([g for g, _ in keys])
    for method, df in preds.items():
        pred = np.zeros((len(ids), len(keys)))
        pred[[position[s] for s in df.sample_id],
             [index[g, a] for g, a in zip(df.gene, df.allele)]] = df.dosage.to_numpy()
        for _, _, name in BINS:
            mask = bins == name
            if not mask.any():
                continue
            rows.append(dict(method=method, maf_bin=name, n_alleles=int(mask.sum()),
                             **micro(pred, truth, mask)))
            for fold in FOLDS:                      # chỉ để đo độ tản, không phải số chính
                sel = fold_vec == fold
                fold_rows.append(dict(method=method, fold=fold, maf_bin=name,
                                      **micro(pred[sel], truth[sel], mask)))
            for gene in LOCI:
                sub = mask & (gene_vec == gene)
                if sub.any():
                    locus_rows.append(dict(method=method, locus=gene, maf_bin=name,
                                           n_alleles=int(sub.sum()),
                                           **micro(pred, truth, sub)))
    summary = pd.DataFrame(rows)
    per_locus = pd.DataFrame(locus_rows)
    per_locus.to_csv(OUT / f"metrics_by_locus_maf_bin{SUFFIX}.csv", index=False)
    per_fold = pd.DataFrame(fold_rows)
    # Trung bình 10 fold (số dùng để VẼ) và độ lệch chuẩn giữa fold, đặt cạnh giá trị
    # gộp toàn cohort để thấy hai cách tổng hợp có lệch nhau không.
    stats = per_fold.groupby(["method", "maf_bin"])[["sn", "ppv", "f1", "r2"]].agg(["mean", "std"])
    for metric in ("sn", "ppv", "f1", "r2"):
        for kind in ("mean", "std"):
            summary[f"{metric}_fold_{kind}"] = [stats.loc[(m, b), (metric, kind)]
                                                for m, b in zip(summary.method, summary.maf_bin)]

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / f"summary_by_maf_bin{SUFFIX}.csv", index=False)
    per_fold.to_csv(OUT / f"metrics_by_fold_maf_bin{SUFFIX}.csv", index=False)
    plot(summary)
    report(summary, per_fold, per_locus)
    print(summary.round(4).to_string(index=False))


def plot(summary):
    """Figure1: cùng bố cục với `do noi bo`
    (một cột mỗi metric, một đường mỗi method), một hàng vì CV chỉ có một ô dữ liệu.

    Giá trị vẽ là giá trị GỘP trên 10 fold (đúng cột trong bảng), không phải trung bình
    fold: giao thức 10-fold chấm trên tập test đã ghép. Hai cách lệch nhau tới 0.0036 ở
    bin `<1%` (base: 0.4888 gộp vs 0.4924 trung bình fold), đủ để lệch một ΔF1 sát ngưỡng.

    Với R² khoảng cách còn lớn hơn: trong một fold chỉ có 100 mẫu test nên chỉ
    ~50/133 allele hiếm thực sự xuất hiện, phần còn lại có phương sai 0 và bị loại khỏi
    Pearson. Trung bình fold vì thế tính trên tập allele dễ hơn hẳn và bị thổi lên
    (0.996 so với 0.888 khi gộp). Vẽ trung bình cho riêng cột này là vẽ sai chỉ số.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    order = [name for _, _, name in BINS]
    methods = list(dict.fromkeys(summary.method))
    metrics = {"sn": ("Sensitivity", "sn"), "ppv": ("PPV", "ppv"),
               "f1": ("F1-score", "f1"), "r2": ("Dosage R²", "r2")}
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6),
                             sharex=True, sharey=True)
    for ax, (title, column) in zip(axes, metrics.values()):
        for method in methods:
            sub = summary[summary.method.eq(method)].set_index("maf_bin").reindex(order)
            ax.plot(order, sub[column], marker="o", linewidth=2.2,
                    markersize=5, color=COLORS.get(method, "#0b0b0b"), label=method)
        ax.set_title(f"VN1K/APMDA — {title}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Allele frequency")
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .95),
               ncol=len(methods), frameon=False)
    fig.suptitle("CV VN1K/APMDA — test 10 fold gộp (locus gộp, 996 mẫu)",
                 y=1.0, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(OUT / f"Figure1{SUFFIX}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"Figure1{SUFFIX}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    assert Image.open(OUT / f"Figure1{SUFFIX}.png").convert("RGBA").getpixel((5, 5)) == (255, 255, 255, 255)


def report(summary, per_fold, per_locus):
    (OUT / f"BAO_CAO_TONG_HOP{SUFFIX}.md").write_text(
        "# CV VN1K/APMDA 10 fold — chấm trên dự đoán gộp\n\n"
        "Mỗi mẫu là test của đúng một fold, nên 10 fold gộp lại là một dự đoán hoàn "
        "chỉnh trên toàn cohort. Chỉ số micro trong từng MAF bin, `TP = min(truth, dosage)`; "
        "tần suất chia bin lấy từ nhãn toàn cohort. `f1_fold_sd` là độ lệch chuẩn của F1 "
        "giữa 10 fold (chỉ để đo độ tản, không phải số chính).\n\n"
        "```\n" + summary.round(4).to_string(index=False) + "\n```\n")
    (OUT / f"BAO_CAO_CHI_TIET{SUFFIX}.md").write_text(
        "# Chi tiết\n\n## Micro theo locus × MAF bin (trên file gộp)\n\n```\n"
        + per_locus.round(4).to_string(index=False)
        + "\n```\n\n## Micro theo fold × MAF bin (đo độ tản)\n\n```\n"
        + per_fold.round(5).to_string(index=False) + "\n```\n")


def self_test():
    """micro phải khớp định nghĩa TP = min(truth, dosage) trên ví dụ tay."""
    truth = np.array([[2., 0.], [1., 1.]])
    pred = np.array([[1.5, .5], [1., 0.]])
    both = micro(pred, truth, np.array([True, True]))
    assert both["true_copies"] == 4 and both["predicted_copies"] == 3
    assert both["tp"] == 1.5 + 0 + 1 + 0
    assert np.isclose(both["f1"], 2 * 2.5 / 7)
    rare = micro(pred, truth, np.array([False, True]))
    assert rare["true_copies"] == 1 and rare["tp"] == 0
    for scheme, cases in (
            ("legacy", ((0., "<1%"), (.015, "1-2%"), (.5, ">=10%"))),
            ("standard", ((.03, "1-5%"), (.15, "10-20%"), (.5, ">=20%"))),
            ("rare", ((0., "AF=0"), (.0005, "0-0.1%"), (.007, "0.5-1%"), (.5, ">=20%")))):
        global BINS, SCHEME
        BINS, SCHEME = BIN_SCHEMES[scheme], scheme
        for freq, name in cases:
            assert bin_name(freq) == name, (scheme, freq, bin_name(freq))
    BINS, SCHEME = BIN_SCHEMES["legacy"], "legacy"
    print("self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pool"); p.add_argument("--method", nargs="+", required=True)
    s = sub.add_parser("score"); s.add_argument("--methods", nargs="+",
                                                default=["SNP2HLA", "CookHLA"])
    sub.add_parser("self-test")
    args = ap.parse_args()
    if args.command == "pool":
        for method in args.method:
            pool(method)
    elif args.command == "score":
        score(args.methods)
    else:
        self_test()

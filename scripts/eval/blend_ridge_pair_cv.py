#!/usr/bin/env python3
"""Trộn log-tuyến tính một arm nền với đầu đọc ridge, β mỗi gene fit leave-one-fold-out.

    s = normalize( exp[(1-β)·log p_nen + β·log p_ridge] )   rồi gọi cứng MAP-diploid.

β là MỘT vô hướng cho mỗi gene nên fit LOFO trên chính dự đoán test của 9 fold còn lại
là sạch (dung lượng 1 tham số), và tránh validation 45 mẫu — xem
`base-memorizes-train-split`: không được fit trên train vì base thuộc lòng train.

Giả dược `--placebo` hoán vị dosage ridge giữa các mẫu trong cùng (fold, gene): giữ
nguyên phân bố biên, phá liên kết mẫu↔điểm. β=0 phải là điểm an toàn của arm.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/eval"), str(ROOT / "scripts/cv_vn1k")]
import common  # noqa: E402

# Cohort khac (HAN) co thu muc du doan rieng; de doi bang env.
PRED = Path(os.environ.get("BLEND_PRED", str(ROOT / "results/cv_vn1k/predictions")))
FOLDS = range(1, 11)
BETA_GRID = tuple(np.round(np.linspace(0, 1, 21), 2))
# luoi tho cho che do hai beta: 11x11 to hop x 9 fold LOFO
COARSE = tuple(np.round(np.linspace(0, 1, 11), 2))


def load(name):
    frame = pd.read_csv(PRED / f"{name}.dosage.csv.gz")
    # ID mau HAN la so nen pandas doc thanh int, con nhan la chuoi -> tra khong ra.
    # Ep ve chuoi ca hai phia thay vi de no im lang ra F1 = 0.
    frame["sample_id"] = frame.sample_id.astype(str)
    return frame


def wide(frame, gene, samples, alleles):
    """(mẫu × allele) dosage, thiếu thì 0."""
    block = frame[frame.gene.eq(gene)]
    return (block.pivot_table(index="sample_id", columns="allele", values="dosage",
                              aggfunc="sum")
            .reindex(index=samples, columns=alleles).fillna(0.0).to_numpy(float))


def to_prob(matrix):
    m = np.clip(matrix, 0.0, None) + 1e-6
    return m / m.sum(1, keepdims=True)


def best_guess(prob):
    """Hai bản sao có dosage dư lớn nhất — ĐÚNG bộ giải mã của cv10_canonical_metrics."""
    out = np.zeros_like(prob)
    work = 2.0 * prob.copy()
    rows = np.arange(len(prob))
    for _ in range(2):
        hit = work.argmax(1)
        out[rows, hit] += 1
        work[rows, hit] -= 1
    return out


def map_diploid(prob, topk=5):
    out = np.zeros_like(prob)
    for i in range(len(prob)):
        top = np.argsort(-prob[i])[:topk]
        logp = np.log(np.maximum(prob[i][top], 1e-12))
        best, arg = -np.inf, (top[0], top[0])
        for u, v in itertools.combinations_with_replacement(range(len(top)), 2):
            value = logp[u] + logp[v] + (np.log(2) if u != v else 0.0)
            if value > best:
                best, arg = value, (top[u], top[v])
        out[i, arg[0]] += 1
        out[i, arg[1]] += 1
    return out


def truth_matrix(labels, samples, gene, alleles):
    index = {a: i for i, a in enumerate(alleles)}
    out = np.zeros((len(samples), len(alleles)))
    for row, sample in enumerate(samples):
        for copy in (1, 2):
            value = labels.at[sample, f"{gene}_{copy}"]
            if isinstance(value, str) and value in index:
                out[row, index[value]] += 1
    return out


def micro_f1(pred, truth, mask=None):
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    t, p = truth.sum(), pred.sum()
    tp = np.minimum(pred, truth).sum()
    sn, ppv = tp / max(t, 1), tp / max(p, 1)
    return 0.0 if sn + ppv == 0 else 2 * sn * ppv / (sn + ppv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="AEHLA_PAIR")
    ap.add_argument("--ridge", default="AEHLA_RIDGE_Z")
    ap.add_argument("--out", required=True, help="tên file trong predictions/")
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--decoder", choices=("map", "best_guess"), default="map")
    ap.add_argument("--af-split", type=float, default=None,
                    help="ngưỡng AF: fit RIÊNG β cho allele hiếm và allele phổ biến")
    ap.add_argument("--emit-soft", action="store_true",
                    help="ghi liều MỀM (2·p) thay vì gọi cứng, để xâu chuỗi hai lần trộn")
    ap.add_argument("--select-bin", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="chọn β theo F1 CHỈ trên allele có AF trong [LO,HI) — dùng khi "
                         "băng cần tối ưu quá nhỏ để lộ ra trong F1 tổng")
    args = ap.parse_args()
    PRED.mkdir(parents=True, exist_ok=True)

    labels = common.load_label(
        os.environ.get("BLEND_COHORT", "VN1K")).map(common.normalize_allele)
    labels.index = labels.index.astype(str)
    protocol = Path(os.environ.get(
        "BLEND_PROTOCOL", str(ROOT / "results/cv_vn1k/protocol")))

    def fold_freq(fold, gene, alleles):
        """AF train của fold, theo gene — cùng định nghĩa CLAUDE.md 2026-08-18."""
        train = (protocol / f"fold{fold:02d}/train.samples").read_text().split()
        values = pd.Series(labels.loc[[s for s in train if s in labels.index],
                                      [f"{gene}_1", f"{gene}_2"]].to_numpy().ravel())
        values = values[values.notna() & values.ne("")]
        counts = values.value_counts()
        total = max(len(values), 1)
        return np.array([counts.get(a, 0) / total for a in alleles])

    base, ridge = load(args.base), load(args.ridge)
    universe = {g: sorted(set(base[base.gene.eq(g)].allele)
                          | set(ridge[ridge.gene.eq(g)].allele)) for g in common.LOCI}
    rng = np.random.default_rng(77)

    # dựng sẵn ma trận từng (fold, gene) rồi mới chọn β — tránh dựng lại 21 lần
    cache = {}
    for fold in FOLDS:
        b, r = base[base.fold.eq(fold)], ridge[ridge.fold.eq(fold)]
        samples = sorted(set(b.sample_id) & set(r.sample_id))
        for gene in common.LOCI:
            alleles = universe[gene]
            pb, pr = (to_prob(wide(b, gene, samples, alleles)),
                      to_prob(wide(r, gene, samples, alleles)))
            if args.placebo:
                pr = pr[rng.permutation(len(pr))]
            cache[(fold, gene)] = (samples, alleles, np.log(pb), np.log(pr),
                                   truth_matrix(labels, samples, gene, alleles),
                                   fold_freq(fold, gene, alleles))

    decode = best_guess if args.decoder == "best_guess" else map_diploid
    grid = ([(b, b) for b in BETA_GRID] if args.af_split is None
            else [(r, c) for r in COARSE for c in COARSE])

    def mix(entry, beta_rare, beta_common):
        _, _, lb, lr, _, freq = entry
        if args.af_split is None:
            weight = beta_rare
        else:
            weight = np.where(freq < args.af_split, beta_rare, beta_common)[None, :]
        return to_prob(np.exp((1 - weight) * lb + weight * lr))

    rows, chosen = [], []
    for fold in FOLDS:
        for gene in common.LOCI:
            scores = []
            for beta_rare, beta_common in grid:
                total = 0.0
                for other in FOLDS:                      # leave-one-fold-out
                    if other == fold:
                        continue
                    entry = cache[(other, gene)]
                    band = None if args.select_bin is None else (
                        (entry[5] >= args.select_bin[0])
                        & (entry[5] < args.select_bin[1]))
                    total += micro_f1(decode(mix(entry, beta_rare, beta_common)),
                                      entry[4], band)
                scores.append(total / 9)
            beta_rare, beta_common = grid[int(np.argmax(scores))]
            chosen.append(dict(fold=fold, gene=gene, beta=float(beta_rare),
                               beta_common=float(beta_common),
                               lofo_f1=float(max(scores))))
            entry = cache[(fold, gene)]
            samples, alleles = entry[0], entry[1]
            mixed = mix(entry, beta_rare, beta_common)
            call = 2.0 * mixed if args.emit_soft else decode(mixed)
            for i, sample in enumerate(samples):
                for a in np.where(call[i] > 1e-4)[0]:
                    rows.append((sample, fold, gene, alleles[a],
                                 float(call[i, a]) if args.emit_soft
                                 else int(call[i, a])))
        print(f"fold{fold:02d} xong", flush=True)

    frame = pd.DataFrame(rows, columns=["sample_id", "fold", "gene", "allele", "dosage"])
    frame.to_csv(PRED / f"{args.out}.dosage.csv.gz", index=False)
    beta_frame = pd.DataFrame(chosen)
    out_dir = ROOT / "results/rare_pathway"
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_frame.to_csv(out_dir / f"beta_{args.out}.csv", index=False)
    print(f"{args.out}: {len(frame)} dòng; β trung vị {beta_frame.beta.median():.2f}, "
          f"β=0 ở {(beta_frame.beta == 0).sum()}/70 ô")


if __name__ == "__main__":
    main()

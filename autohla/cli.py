"""Hai lenh: `autohla train` va `autohla impute`. Xem spec §4.

`impute` KHONG nhan them tham so nao ngoai --model/--vcf/--out: marker, allele
universe, cau hinh va he so tron deu doc tu `model/`. Doi marker sau khi train la
doi mo hinh, nen `--marker-list` chi ton tai o `train`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from.config import RunConfig
from.decode import map_diploid_pairs, to_prob
from.io.dataset import load_dataset
from.io.labels import load_labels
from.io.model_dir import load_model, save_model
from.io.vcf import GROUPS, markers_from_vcf, read_markers, scan_vcf
from.model.ridge import apply_ridge, select_lambda, zscore
from.train.blend import blend, fit_beta
from.train.curriculum import train_curriculum
from.train.pretrain import pretrain_s1
from.train.splits import kfold_by_sample

IMPUTE_COLUMNS = ["sample_id", "gene", "allele_1", "allele_2", "posterior"]
# Ty le val cat ra tu train, quy tac dung cua repo (CLAUDE.md): 5% cua phan train.
VAL_FRACTION = 0.05
# Overlap marker toi thieu giua VCF test va markers.tsv. Duoi nguong nay mo hinh
# dang nhin mot chip khac, va ket qua se la rac IM LANG chu khong phai loi.
MIN_MARKER_OVERLAP = 0.5


def write_calls(path, rows) -> None:
    """Mot dong moi (mau x gene), dung IMPUTE_COLUMNS, khong co dinh dang thay the."""
    import pandas as pd
    pd.DataFrame(list(rows), columns=IMPUTE_COLUMNS).to_csv(path, index=False)


def _sample_ids(vcf_path):
    """ID mau (khong phai hang haplotype) theo dung thu tu on dinh."""
    ids = scan_vcf(vcf_path)["sample_ids"]
    return sorted({i[:-2] if i.endswith(("_1", "_2")) else i for i in ids})


def _holdout(sample_ids, fraction, seed):
    """Cat val ra khoi train. Dung permutation cua splits.kfold_by_sample de hai
    duong (val noi bo va k-fold) khong cho ra hai thu tu mau khac nhau."""
    k = max(2, int(round(1 / fraction)))
    train, val = kfold_by_sample(sample_ids, k=k, seed=seed)[0]
    return train, val


def _gene_slices(outputs_size):
    out, start = [], 0
    for name, size in outputs_size:
        out.append((name.replace("HLA_", ""), slice(start, start + size), size))
        start += size
    return out


def _scores_and_z(net, data):
    x = torch.as_tensor(data, dtype=torch.float32)
    with torch.no_grad:
        return net(x).numpy, net.encode(x).numpy


def _truth_dosage(dataset):
    """Lieu THAT 0/1/2 tu nhan multi-hot da collapse (2 khi dong hop)."""
    return np.asarray(dataset["label"], dtype=float)


def _oof_arms(cfg, vcf, labels, markers, s1_path, work, folds, epochs):
    """Du doan OUT-OF-FOLD cua ca hai arm tren toan bo mau train.

    Tra ({gene: prob_nen}, {gene: prob_ridge}, {gene: truth}, {gene: af}) --
    dung dang `fit_beta` can. Moi fold huan luyen lai tu dau tren 9/10 con lai,
    do la dieu lam du doan thuc su out-of-fold: mo hinh nen THUOC LONG train
    (do noi bo), fit he so tron tren train cho beta = 0.
    """
    ids = _sample_ids(vcf)
    base, extra, truth, gene_size = {}, {}, {}, {}
    for i, (fold_train, fold_test) in enumerate(
            kfold_by_sample(ids, k=folds, seed=cfg.seed), start=1):
        inner_train, inner_val = _holdout(fold_train, VAL_FRACTION, cfg.seed)
        net = train_curriculum(cfg, vcf, vcf, labels, markers, s1_path,
                               str(Path(work) / f"oof{i:02d}"), epochs=epochs,
                               train_samples=inner_train, val_samples=inner_val)
        net.eval
        parts = {}
        for split, keep in (("train", inner_train), ("val", inner_val),
                            ("test", fold_test)):
            ds = load_dataset(vcf, labels, markers, cfg.group, 4, "test",
                              cfg.phased, keep_samples=keep)
            score, z = _scores_and_z(net, ds["data"])
            parts[split] = (score, z, _truth_dosage(ds))
        outputs_size = net.outputs_size
        for gene, sl, size in _gene_slices(outputs_size):
            gene_size[gene] = size
            tr_z, va_z, te_z = zscore(parts["train"][1], parts["val"][1],
                                      parts["test"][1])
            ytr = (parts["train"][2][:, sl] > 0).astype(float)
            yva = (parts["val"][2][:, sl] > 0).astype(float)
            w, _ = select_lambda(tr_z, ytr, va_z, yva)
            base.setdefault(gene, []).append(to_prob(parts["test"][0][:, sl]))
            extra.setdefault(gene, []).append(to_prob(apply_ridge(te_z, w)))
            truth.setdefault(gene, []).append(parts["test"][2][:, sl])
    stack = lambda d: {g: np.concatenate(v, axis=0) for g, v in d.items}  # noqa: E731
    base, extra, truth = stack(base), stack(extra), stack(truth)
    # AF theo CLAUDE.md: ban sao allele / tong ban sao HLA hop le CUA CHINH gene do.
    freq = {g: truth[g].sum(0) / max(truth[g].sum, 1.0) for g in truth}
    return base, extra, truth, freq


def _fit_final_ridge(net, cfg, vcf, labels, markers, train_ids, val_ids):
    """Ridge CUOI CUNG di kem model/: fit tren z cua toan bo train, lam chon tren
    val. Mot ma tran trong so moi gene."""
    parts = {}
    for split, keep in (("train", train_ids), ("val", val_ids)):
        ds = load_dataset(vcf, labels, markers, cfg.group, 4, "test", cfg.phased,
                          keep_samples=keep)
        _, z = _scores_and_z(net, ds["data"])
        parts[split] = (z, _truth_dosage(ds))
    tr_z, va_z = zscore(parts["train"][0], parts["val"][0])
    weights = {}
    for gene, sl, _ in _gene_slices(net.outputs_size):
        ytr = (parts["train"][1][:, sl] > 0).astype(float)
        yva = (parts["val"][1][:, sl] > 0).astype(float)
        weights[gene], _ = select_lambda(tr_z, ytr, va_z, yva)
    return weights, parts["train"][0].mean(0), parts["train"][0].std(0) + 1e-8


def _train(args) -> int:
    cfg = RunConfig.from_vcf(args.vcf, args.group, phase=args.phase,
                             marker_list=args.marker_list, head=args.head)
    print(cfg.explain)
    markers = (read_markers(args.marker_list) if args.marker_list
               else markers_from_vcf(args.vcf))
    print("markers      = {} (frozen into model/markers.tsv)".format(len(markers)))
    labels = load_labels(args.labels, GROUPS[args.group], n_digits=4)

    out = Path(args.out)
    work = out / "work"
    ids = _sample_ids(args.vcf)
    train_ids, val_ids = _holdout(ids, VAL_FRACTION, cfg.seed)

    s1_path = None
    if not args.no_s1:
        s1_path = str(work / "s1.pt")
        work.mkdir(parents=True, exist_ok=True)
        pretrain_s1(args.vcf, args.vcf, markers, args.group, s1_path,
                    epochs=args.s1_epochs, threads=args.threads, head=cfg.head)

    net = train_curriculum(cfg, args.vcf, args.vcf, labels, markers, s1_path,
                           str(work / "main"), epochs=args.epochs,
                           train_samples=train_ids, val_samples=val_ids)
    net.eval

    beta = ridge = None
    if cfg.use_ridge and not args.no_posthoc:
        base, extra, truth, freq = _oof_arms(cfg, args.vcf, labels, markers,
                                             s1_path, work, args.posthoc_folds,
                                             args.epochs)
        beta = fit_beta(base, extra, truth, freq, af_split=args.af_split)
        print("beta         = " + ", ".join(
            "{} {:.2f}/{:.2f}".format(g, r, c) for g, (r, c) in sorted(beta.items)))
        weights, mu, sd = _fit_final_ridge(net, cfg, args.vcf, labels, markers,
                                           train_ids, val_ids)
        ridge = dict(weights, **{"__mu": mu, "__sd": sd})

    ds = load_dataset(args.vcf, labels, markers, args.group, 4, "test", cfg.phased,
                      keep_samples=train_ids)
    save_model(out, net, cfg, markers, ds["encoder"], beta=beta, ridge=ridge,
               af_split=args.af_split)
    print("written      = {}".format(out))
    return 0


def _impute(args) -> int:
    model = load_model(args.model)
    markers = model["markers"]
    present = set(markers_from_vcf(args.vcf))
    overlap = len(present & set(markers)) / max(len(markers), 1)
    if overlap < MIN_MARKER_OVERLAP:
        raise SystemExit(
            "the VCF covers only {:.1%} of the {} markers in model/ ({} missing). "
            "This is almost certainly a different chip and the calls would be "
            "garbage. Stopping.".format(overlap, len(markers), len(set(markers) - present)))

    net, cfg = model["net"], model["manifest"]["config"]
    ds = load_dataset(args.vcf, None, markers, cfg["group"], 4, "unlabeled",
                      cfg["phased"])
    ds["encoder"] = model["encoder"]
    score, z = _scores_and_z(net, ds["data"])

    beta, ridge = model["beta"], model["ridge"]
    decoder = model["encoder"].decoder
    rows = []
    for gene, sl, _ in _gene_slices(net.outputs_size):
        prob = to_prob(score[:, sl])
        if beta is not None and ridge is not None and gene in ridge:
            zz = (z - ridge["__mu"]) / ridge["__sd"]
            extra = to_prob(apply_ridge(zz, ridge[gene]))
            freq = prob.mean(0)          # khong co nhan test -> AF uoc tu chinh du doan
            prob = blend(prob, extra, beta[gene][0], beta[gene][1], freq,
                         af_split=model["manifest"]["af_split"])
        pairs, posterior = map_diploid_pairs(prob)
        for i, sample in enumerate(ds["sample-list"]):
            rows.append((str(sample), gene, decoder[(gene, int(pairs[i, 0]))],
                         decoder[(gene, int(pairs[i, 1]))], float(posterior[i])))
    rows.sort(key=lambda r: (r[0], r[1]))
    write_calls(args.out, rows)
    print("{} rows -> {}".format(len(rows), args.out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autohla")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train from a VCF + labels, write model/")
    t.add_argument("--vcf", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--group", required=True, type=int, choices=sorted(GROUPS))
    t.add_argument("--out", required=True)
    t.add_argument("--marker-list", default=None,
                   help="chip marker list; inferred from the training VCF if omitted")
    t.add_argument("--phase", default="auto", choices=("auto", "on", "off"),
                   help="phasing is OPT-IN: both auto and off leave it OFF (see config.py)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--s1-epochs", type=int, default=100)
    t.add_argument("--no-s1", action="store_true",
                   help="skip the S1 pretext stage (measured null on downstream F1)")
    t.add_argument("--no-posthoc", action="store_true",
                   help="skip the ridge + blend layer (several times faster)")
    t.add_argument("--posthoc-folds", type=int, default=10)
    t.add_argument("--af-split", type=float, default=0.20)
    t.add_argument("--head", default="full", choices=("full", "lean"),
                   help="lean = drop fc1/fc2 so fc3 reads z directly (-87%% readout parameters)")
    t.set_defaults(func=_train)

    i = sub.add_parser("impute", help="call alleles for a new VCF with an existing model/")
    i.add_argument("--model", required=True)
    i.add_argument("--vcf", required=True)
    i.add_argument("--out", required=True)
    i.set_defaults(func=_impute)

    for parser in (t, i):
        parser.add_argument("--threads", type=int, default=8,
                            help="torch threads. Default 8 -- left alone, torch "
                                 "takes every core on the machine")

    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main)

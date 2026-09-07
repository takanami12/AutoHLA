"""Hai lenh: `autohla train` va `autohla impute`. Xem spec §4.

`impute` KHONG nhan them tham so nao ngoai --model/--vcf/--out: marker, allele
universe, cau hinh va he so tron deu doc tu `model/`. Doi marker sau khi train la
doi mo hinh, nen `--marker-list` chi ton tai o `train`.
"""
import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import PAIR_SKIP_N, RunConfig
from .decode import HW_TAU, map_diploid_pairs, to_prob, tune_tau
from .io.dataset import allele_frequencies, load_dataset
from .io.labels import load_labels
from .io.model_dir import load_model, save_model
from .io.vcf import GROUPS, markers_from_vcf, read_markers, scan_vcf
from .model.pair import PairEnergyHead
from .model.ridge import apply_ridge, ridge_prob, select_lambda, zscore
from .train.blend import blend, fit_beta
from .train.curriculum import train_curriculum
from .train.pair import predict_pair, train_pair
from .train.pretrain import pretrain_s1
from .train.splits import kfold_by_sample

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
    if len(ids) != len(set(ids)):
        raise ValueError("VCF sample IDs must be unique")
    return sorted(ids)


def _holdout(sample_ids, fraction, seed):
    """Cat val ra khoi train. Dung permutation cua splits.kfold_by_sample de hai
    duong (val noi bo va k-fold) khong cho ra hai thu tu mau khac nhau."""
    k = min(len(sample_ids), max(2, int(round(1 / fraction))))
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
    with torch.no_grad():
        score, z = net.forward_with_embedding(x)
        return score.numpy(), z.numpy()


def _blended_probs(net, score, z, beta, ridge, af_split, pair=None, x=None,
                   train_af=None) -> dict:
    """{gene: phan bo xac suat} sau khi tron ridge -- DUONG DUY NHAT sinh xac suat
    dem di giai ma. `_train` fit tau tren chinh ham nay va `_impute` goi lai no,
    nen tau khong bao gio duoc chon tren mot phan bo khac voi phan bo luc impute.

    `pair` = {"net": AutoNet da fine-tune, "head": PairEnergyHead} va `x` la dau
    vao tho de chay no. Co pair thi arm NEN la dosage cua pair chu khong phai
    `score` -- dung nhu blend_ridge_pair_cv, noi arm nen la file *_PAIR. Arm phu
    (ridge) van doc `z` cua trunk NEN, khong phai cua trunk da fine-tune.

    AF is frozen from reference labels during train, never estimated from test.
    """
    base = score
    if pair is not None:
        if x is None:
            raise ValueError("pair needs x: the fine-tuned trunk has to be run")
        base = predict_pair(pair["net"], pair["head"],
                            torch.as_tensor(x, dtype=torch.float32))
    out = {}
    for gene, sl, _ in _gene_slices(net.outputs_size):
        prob = to_prob(base[:, sl])
        if beta is not None and ridge is not None and gene in ridge:
            if af_split is not None and (train_af is None or gene not in train_af):
                raise ValueError("AF-split ridge needs frozen train_af; retrain this model")
            zz = (z - ridge["__mu"]) / ridge["__sd"]
            extra = ridge_prob(apply_ridge(zz, ridge[gene]))
            prob = blend(prob, extra, beta[gene][0], beta[gene][1],
                         None if train_af is None else train_af[gene],
                         af_split=af_split)
        out[gene] = prob
    return out


def _truth_dosage(dataset):
    """True copy counts, separate from the binary BCE target."""
    return np.asarray(dataset["dosage"], dtype=float)


def _pair_dosage(net, cfg, parts, epochs, log=print):
    """Fine-tune mot BAN SAO cua `net` cung PairEnergyHead, tra dosage tren test.

    `net` KHONG bi doi: arm ridge doc `z` cua chinh no, va `save_model` ghi no ra
    lam trunk nen. train_pair fine-tune tai cho nen ban sao la bat buoc.

    `parts[split]` la (score, z, truth, x) -- chi truth va x duoc dung o day.
    """
    sizes = [size for _, _, size in _gene_slices(net.outputs_size)]
    torch.manual_seed(cfg.seed)
    head = PairEnergyHead(cfg.shared_dim, sizes)
    pair_net, head, _ = train_pair(
        copy.deepcopy(net), head, parts["train"][3],
        torch.as_tensor(parts["train"][2] > 0, dtype=torch.float32), parts["train"][2],
        parts["val"][3], parts["val"][2], epochs=epochs, log=log,
        train_mask=parts["train"][4] if len(parts["train"]) > 4 else None,
        val_valid_copies=parts["val"][5] if len(parts["val"]) > 5 else None)
    return predict_pair(pair_net, head, parts["test"][3]), pair_net, head


def _parts(net, dataset):
    score, z = _scores_and_z(net, dataset["data"])
    return (score, z, _truth_dosage(dataset),
            torch.as_tensor(dataset["data"], dtype=torch.float32),
            dataset["label-mask"], dataset["valid-copies"])


def _reference_af(dataset):
    sizes = [size for _, size in dataset["outputs-size"]]
    af = allele_frequencies(dataset["dosage"], sizes)
    return {gene: af[sl] for gene, sl, _ in _gene_slices(dataset["outputs-size"])}


def _ridge_weights(train_z, val_z, train, val, sl, gene_index):
    train_ok = train[4][:, sl].all(1)
    val_ok = val[5][:, gene_index] == 2
    if not train_ok.any():
        raise ValueError("ridge needs at least one complete training genotype per gene")
    return select_lambda(train_z[train_ok], (train[2][train_ok, sl] > 0).astype(float),
                         val_z[val_ok], (val[2][val_ok, sl] > 0).astype(float),
                         val_dosage=val[2][val_ok, sl],
                         val_valid_copies=val[5][val_ok, gene_index])[0]


def _oof_arms(cfg, vcf, labels, markers, s1_path, work, folds, epochs,
              net_source=None, *, pair_epochs=None, gene_weights=None,
              reference_ids=None, encoder=None, s1_epochs=100, threads=2,
              return_valid=False):
    """Du doan OUT-OF-FOLD cua ca hai arm tren toan bo mau train.

    Tra ({gene: prob_nen}, {gene: prob_ridge}, {gene: truth}, {gene: af}) --
    dung dang `fit_beta` can. Moi fold huan luyen lai tu dau tren 9/10 con lai,
    do la dieu lam du doan thuc su out-of-fold: mo hinh nen THUOC LONG train
    (memory `base-memorizes-train-split`), fit he so tron tren train cho beta = 0.

    `net_source(i, inner_train, inner_val)` thay cho viec huan luyen lai fold thu
    i. `tools/refit_posthoc.py` truyen mot ham nap lai checkpoint da nam san trong
    `work/oofNN/`, de tinh lai tang hau ky sau mot ban vá ma khong phai tra 11x
    lan nua. None = huan luyen that, duong mac dinh. Diem quan trong: hai duong
    dung CHUNG than ham nay, nen ban refit khong the lech khoi ban train -- do
    dung la kieu lech da sinh ra bug kep ridge (fit_posthoc_cv duoc vá, cli thi
    khong).

    `pair_epochs` khac None thi arm NEN doi tu `to_prob(score)` sang dosage cua
    tang pair, dung nhu tools/fit_posthoc_cv.py: o do arm nen la file *_PAIR chu
    khong phai du doan tho. None = giu dung hanh vi cu, nen `refit_posthoc.py`
    goi positional van chay.
    """
    ids = _sample_ids(vcf) if reference_ids is None else list(reference_ids)
    reference = load_dataset(vcf, labels, markers, cfg.group, 4, "train",
                             cfg.phased, keep_samples=ids, encoder=encoder)
    encoder = reference["encoder"]
    base, extra, truth, valid = {}, {}, {}, {}
    for i, (fold_train, fold_test) in enumerate(
            kfold_by_sample(ids, k=folds, seed=cfg.seed), start=1):
        inner_train, inner_val = _holdout(fold_train, VAL_FRACTION, cfg.seed)
        inner_work = Path(work) / f"oof{i:02d}"
        inner_s1 = None
        if s1_path is not None and net_source is None:
            inner_work.mkdir(parents=True, exist_ok=True)
            inner_s1 = str(inner_work / "s1.pt")
            pretrain_s1(vcf, vcf, markers, cfg.group, inner_s1, epochs=s1_epochs,
                        seed=cfg.seed, threads=threads, head=cfg.head, strides=cfg.strides,
                        train_samples=inner_train, val_samples=inner_val)
        net = (net_source(i, inner_train, inner_val) if net_source is not None
               else train_curriculum(replace(cfg, n_train=len(inner_train)),
                                     vcf, vcf, labels, markers, inner_s1,
                                     str(inner_work), epochs=epochs,
                                     train_samples=inner_train, val_samples=inner_val,
                                     gene_weights=gene_weights))
        net.eval()
        parts = {}
        for split, keep in (("train", inner_train), ("val", inner_val),
                            ("test", fold_test)):
            ds = load_dataset(vcf, labels, markers, cfg.group, 4, "test",
                              cfg.phased, keep_samples=keep, encoder=net.encoder)
            parts[split] = _parts(net, ds)
        outer_test = load_dataset(vcf, labels, markers, cfg.group, 4, "test",
                                  cfg.phased, keep_samples=fold_test, encoder=encoder)
        pair_test = None
        if pair_epochs:
            pair_test, _, _ = _pair_dosage(
                net, cfg, parts, pair_epochs,
                log=lambda m, i=i: print("oof{:02d} {}".format(i, m), flush=True))
        outputs_size = net.outputs_size
        outer_slices = {g: sl for g, sl, _ in _gene_slices(outer_test["outputs-size"])}
        tr_z, va_z, te_z = zscore(parts["train"][1], parts["val"][1], parts["test"][1])
        for gi, (gene, sl, size) in enumerate(_gene_slices(outputs_size)):
            w = _ridge_weights(tr_z, va_z, parts["train"], parts["val"], sl, gi)
            columns = {encoder.decoder[(gene, j)]: j
                       for j in range(encoder.label_counter[gene])}
            target = [columns[net.encoder.decoder[(gene, j)]] for j in range(size)]
            def align(values):
                aligned = np.zeros((len(fold_test), len(columns)))
                aligned[:, target] = values
                return aligned
            base.setdefault(gene, []).append(align(to_prob(
                parts["test"][0][:, sl] if pair_test is None else pair_test[:, sl])))
            extra.setdefault(gene, []).append(align(ridge_prob(apply_ridge(te_z, w))))
            truth.setdefault(gene, []).append(outer_test["dosage"][:, outer_slices[gene]])
            valid.setdefault(gene, []).append(outer_test["valid-copies"][:, gi])
    stack = lambda d: {g: np.concatenate(v, axis=0) for g, v in d.items()}  # noqa: E731
    base, extra, truth = stack(base), stack(extra), stack(truth)
    result = (base, extra, truth, _reference_af(reference))
    return result + (stack(valid),) if return_valid else result


def _fit_final_ridge(net, cfg, vcf, labels, markers, train_ids, val_ids, val_vcf=None):
    """Ridge CUOI CUNG di kem model/: fit tren z cua toan bo train, lam chon tren
    val. Mot ma tran trong so moi gene."""
    parts = {}
    for split, keep, path in (("train", train_ids, vcf), ("val", val_ids, val_vcf or vcf)):
        ds = load_dataset(path, labels, markers, cfg.group, 4, "test", cfg.phased,
                          keep_samples=keep, encoder=net.encoder)
        parts[split] = _parts(net, ds)
    tr_z, va_z = zscore(parts["train"][1], parts["val"][1])
    weights = {}
    for gi, (gene, sl, _) in enumerate(_gene_slices(net.outputs_size)):
        weights[gene] = _ridge_weights(tr_z, va_z, parts["train"], parts["val"], sl, gi)
    return weights, parts["train"][1].mean(0), parts["train"][1].std(0) + 1e-8


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_training_provenance(out, args, labels, train_ids, val_ids):
    """Never resume a checkpoint from different data, code, or training options."""
    source = Path(__file__).parent
    options = {key: value for key, value in vars(args).items()
               if key not in {"func", "out", "vcf", "val_vcf", "labels", "marker_list"}}
    record = {
        "options": options, "train_ids": train_ids, "val_ids": val_ids,
        "inputs": {key: _file_hash(getattr(args, key)) for key in
                   ("vcf", "val_vcf", "marker_list") if getattr(args, key, None)},
        "labels_sha256": hashlib.sha256(labels.loc[sorted(train_ids + val_ids)]
                                         .to_csv().encode()).hexdigest(),
        "source": {str(path.relative_to(source)): _file_hash(path)
                   for path in sorted(source.rglob("*.py"))},
    }
    path = out / "training_provenance.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("training provenance mismatch; use a new --out directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("non-empty --out lacks training provenance; use a new directory")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _train(args) -> int:
    if min(args.epochs, args.s1_epochs, args.pair_epochs) < 1:
        raise ValueError("epoch counts must be positive")
    if args.posthoc_folds < 2:
        raise ValueError("--posthoc-folds must be at least 2")
    if not 0 < args.af_split <= 1:
        raise ValueError("--af-split must be in (0,1]")
    force_pair = {"auto": None, "on": True, "off": False}[args.pair]
    strides = tuple(int(x) for x in args.strides.split(","))
    if len(strides) != 2 or any(s < 1 for s in strides):
        raise SystemExit("--strides can dung dang 'a,b' voi a,b >= 1, nhan {!r}"
                         .format(args.strides))
    gene_weights = {}
    for item in filter(None, args.gene_loss_weight.split(",")):
        name, _, value = item.partition("=")
        gene_weights["HLA_" + name.replace("HLA_", "")] = float(value)
    if any(not np.isfinite(v) or v <= 0 for v in gene_weights.values()):
        raise ValueError("gene loss weights must be finite and positive")
    cfg = RunConfig.from_vcf(args.vcf, args.group, phase=args.phase,
                             marker_list=args.marker_list, head=args.head,
                             force_pair=force_pair, strides=strides)
    markers = read_markers(args.marker_list)
    print("markers      = {} (frozen into model/markers.tsv)".format(len(markers)))
    labels = load_labels(args.labels, GROUPS[args.group], n_digits=4)

    out = Path(args.out)
    work = out / "work"
    ids = _sample_ids(args.vcf)
    val_vcf = args.val_vcf or args.vcf
    if args.val_vcf:
        train_ids, val_ids = ids, _sample_ids(args.val_vcf)
        if set(train_ids) & set(val_ids):
            raise ValueError("train and validation sample IDs overlap")
    else:
        train_ids, val_ids = _holdout(ids, VAL_FRACTION, cfg.seed)
    if len(train_ids) < 2 or not val_ids:
        raise ValueError("need at least two training samples and one validation sample")
    if not labels.index.is_unique:
        raise ValueError("label sample IDs must be unique")
    missing = set(train_ids + val_ids) - set(labels.index)
    if missing:
        raise ValueError(f"{len(missing)} train/validation samples missing from labels")
    labels = labels.loc[sorted(train_ids + val_ids)]
    cfg = replace(cfg, n_train=len(train_ids), use_ridge=len(train_ids) < PAIR_SKIP_N,
                  use_pair=len(train_ids) < PAIR_SKIP_N if force_pair is None else force_pair)
    print(cfg.explain())
    provenance = _check_training_provenance(out, args, labels, train_ids, val_ids)

    s1_path = None
    if not args.no_s1:
        s1_path = str(work / "s1.pt")
        work.mkdir(parents=True, exist_ok=True)
        pretrain_s1(args.vcf, val_vcf, markers, args.group, s1_path,
                    epochs=args.s1_epochs, threads=args.threads, head=cfg.head,
                    strides=cfg.strides, train_samples=train_ids, val_samples=val_ids)

    if gene_weights:
        print("gene_weights = " + ", ".join(
            "{} {:.2f}".format(g, w) for g, w in sorted(gene_weights.items())))
    net = train_curriculum(cfg, args.vcf, val_vcf, labels, markers, s1_path,
                           str(work / "main"), epochs=args.epochs,
                           train_samples=train_ids, val_samples=val_ids,
                           gene_weights=gene_weights or None)
    net.eval()
    train_ds = load_dataset(args.vcf, labels, markers, args.group, 4, "train", cfg.phased,
                             keep_samples=train_ids, encoder=net.encoder)
    freq = _reference_af(train_ds)

    beta = ridge = pair = pair_net = None
    calibration = None
    tau = HW_TAU
    use_pair = cfg.use_pair and not args.no_posthoc
    if cfg.use_ridge and not args.no_posthoc:
        base, extra, truth, freq, valid = _oof_arms(
            cfg, args.vcf, labels, markers, s1_path, work, args.posthoc_folds,
            args.epochs, pair_epochs=args.pair_epochs if use_pair else None,
            gene_weights=gene_weights or None, reference_ids=train_ids,
            encoder=net.encoder, s1_epochs=args.s1_epochs, threads=args.threads,
            return_valid=True)
        complete = {g: valid[g] == 2 for g in valid}
        beta = fit_beta({g: p[complete[g]] for g, p in base.items()},
                        {g: p[complete[g]] for g, p in extra.items()},
                        {g: p[complete[g]] for g, p in truth.items()}, freq,
                        af_split=args.af_split)
        calibration = [(blend(base[g], extra[g], *beta[g], freq[g],
                              af_split=args.af_split), truth[g], valid[g]) for g in base]
        print("beta         = " + ", ".join(
            "{} {:.2f}/{:.2f}".format(g, r, c) for g, (r, c) in sorted(beta.items())))
        # Ridge TRUOC pair: no phai duoc fit tren z cua trunk NEN. Doi thu tu la
        # doi dai luong, va hai tang thoi doc lap (fit_posthoc_cv.py cung the).
        weights, mu, sd = _fit_final_ridge(net, cfg, args.vcf, labels, markers,
                                           train_ids, val_ids, val_vcf=val_vcf)
        ridge = dict(weights, **{"__mu": mu, "__sd": sd})

    val_ds = load_dataset(val_vcf, labels, markers, args.group, 4, "test",
                          cfg.phased, keep_samples=val_ids, encoder=net.encoder)
    val_truth = _truth_dosage(val_ds)
    if use_pair:
        val_parts = _parts(net, val_ds)
        parts = {"train": _parts(net, train_ds), "val": val_parts, "test": val_parts}
        _, pair_net, head = _pair_dosage(net, cfg, parts, args.pair_epochs)
        pair = {"net": pair_net, "head": head}
        print("pair         = fine-tuned trunk + {} epochs".format(args.pair_epochs))

    calibration_source = "oof" if calibration is not None else "validation"
    if calibration is None:
        val_score, val_z = _scores_and_z(net, val_ds["data"])
        val_prob = _blended_probs(net, val_score, val_z, beta, ridge, args.af_split,
                                  pair=pair, x=val_ds["data"], train_af=freq)
        calibration = [(val_prob[gene], val_truth[:, sl], val_ds["valid-copies"][:, gi])
                       for gi, (gene, sl, _) in enumerate(_gene_slices(net.outputs_size))]
    tau = tune_tau(calibration, prefer_hw=True)
    provenance["calibration_source"] = calibration_source
    print(f"tau          = {tau:.3f} ({calibration_source}; {HW_TAU} = Hardy-Weinberg)")

    save_model(out, net, cfg, markers, net.encoder, beta=beta, ridge=ridge,
               pair=None if pair is None else pair["head"].state_dict(),
               pair_net=pair_net, af_split=args.af_split, tau=tau,
               train_af=freq, training_provenance=provenance)
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
            "garbage. Stopping."
            .format(overlap, len(markers), len(set(markers) - present)))

    net, cfg = model["net"], model["manifest"]["config"]
    ds = load_dataset(args.vcf, None, markers, cfg["group"], 4, "unlabeled",
                      cfg["phased"])
    ds["encoder"] = model["encoder"]
    if not len(ds["sample-list"]):
        raise ValueError("VCF contains no samples")
    if np.any(np.all(ds["data"][:, 2, :] > 0, axis=1)):
        raise ValueError("VCF contains samples with all model markers missing")
    score, z = _scores_and_z(net, ds["data"])

    beta, ridge = model["beta"], model["ridge"]
    decoder = model["encoder"].decoder
    rows = []
    # tau vang mat = model/ cu, truoc khi vach dong hop duoc fit -> Hardy-Weinberg.
    tau = model["manifest"].get("tau", HW_TAU)
    probs = _blended_probs(net, score, z, beta, ridge,
                           model["manifest"]["af_split"],
                           pair=model["pair"], x=ds["data"], train_af=model["train_af"])
    for gene, prob in probs.items():
        pairs, posterior = map_diploid_pairs(prob, tau=tau)
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
    t.add_argument("--val-vcf", help="fixed validation split; avoids an additional internal split")
    t.add_argument("--labels", required=True)
    t.add_argument("--group", required=True, type=int, choices=sorted(GROUPS))
    t.add_argument("--out", required=True)
    t.add_argument("--marker-list", required=True,
                   help="reference chip marker allowlist, frozen into the model")
    t.add_argument("--phase", default="auto", choices=("auto", "on", "off"),
                   help="phasing is OPT-IN: both auto and off leave it OFF (see config.py)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--s1-epochs", type=int, default=100)
    t.add_argument("--no-s1", action="store_true",
                   help="skip the S1 pretext stage (measured null on downstream F1)")
    t.add_argument("--no-posthoc", action="store_true",
                   help="skip pair + ridge + blend (several times faster)")
    t.add_argument("--posthoc-folds", type=int, default=10)
    t.add_argument("--pair", default="on", choices=("auto", "on", "off"),
                   help="tang pair (cham diem cap khong thu tu). auto = luat theo "
                        "co train co san trong config (n < PAIR_SKIP_N)")
    t.add_argument("--pair-epochs", type=int, default=40,
                   help="tran epoch fine-tune cua tang pair; khop fit_posthoc_cv.py")
    t.add_argument("--af-split", type=float, default=0.20)
    t.add_argument("--head", default="full", choices=("full", "lean"),
                   help="lean = drop fc1/fc2 so fc3 reads z directly (-87%% readout parameters)")
    t.add_argument("--strides", default="2,2",
                   help="ha mau cua trunk, dang 'a,b'. Mac dinh 2,2. Luoi marker "
                        "thua (vd HAN g4: 333 marker) thi 4x downsample co the vut "
                        "phan giai; S1 va S2 luon dung CUNG gia tri nay")
    t.add_argument("--gene-loss-weight", default="",
                   help="ha trong so mot gene trong loss, dang 'DQA1=0.25[,DRB1=0.5]'. "
                        "Gene VAN co head va van duoc du doan -- chi bot keo trunk "
                        "dung chung, nen khong phai train rieng mot mo hinh cho no")
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
    if args.threads < 1:
        ap.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

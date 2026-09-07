"""Nap VCF + nhan thanh tensor huan luyen.

Port cua AEHLA src/preprocess_data.load_dataset (dong 59-223), rut gon: bo cac
nhanh thi nghiem da bi bac bo (AE_NT, AE_LPHASE_RAND/_SWITCH/_PRED, AE_KD,
AE_HAPREC*, AE_HIER_*, AE_MGDA, AE_S2_CORRUPT, AE_TRUNK*, AE_RESNET_*,
AE_DEC_*, AE_FREEZE_TRUNK, SSL_*, GENE_CNN_*, va nhanh not_collapsed=True cua
kien truc CoNet cu -- HLA_ARCH=ae luon dung nhanh collapsed). Kenh missing
luon bat, giong HLA_MISSING mac dinh cua HLA_ARCH=ae. AE_LPHASE=1 +
AE_LPHASE_ORACLE=1 gop lai thanh mot tham so bool duy nhat: `phased`.
"""
import warnings

import numpy as np
import pandas as pd

from autohla.io.vcf import GROUPS, load_haplotypes


def _valid_allele(value):
    return isinstance(value, str) and value.strip().lower() not in {
        "", ".", "-", "0", "na", "n/a", "nan", "none"}


def allele_frequencies(dosage, sizes):
    """Train/reference copy AF; each gene excludes its own missing copies."""
    dosage = np.asarray(dosage, dtype=float)
    if dosage.ndim != 2 or sum(sizes) != dosage.shape[1]:
        raise ValueError("dosage columns must match gene sizes")
    if not np.isfinite(dosage).all() or (dosage < 0).any():
        raise ValueError("dosage must be finite and non-negative")
    af, start = np.zeros(dosage.shape[1]), 0
    for size in sizes:
        copies = dosage[:, start:start + size].sum(0)
        af[start:start + size] = copies / max(copies.sum(), 1)
        start += size
    return af


def _split_missing(hap_1, hap_2):
    """(-1 sentinel tu load_haplotypes) -> haplotype sach + kenh missing."""
    missing = ((hap_1 < 0) | (hap_2 < 0)) * 1
    return np.where(missing, 0, hap_1), np.where(missing, 0, hap_2), missing


class _Encoder:
    """Ban rut gon cua AEHLA objects.encoder.Encoder -- chi giu phan
    load_dataset can: ma hoa allele -> one-hot theo tung gene, va giai ma nguoc."""

    def __init__(self):
        self.encoder = {}
        self.decoder = {}
        self.label_counter = {}

    def make(self, label_df, columns):
        # Khong nhan n_digits: label_df phai da duoc cat do phan giai roi (xem
        # io/labels.load_labels), nen ma hoa lai o day se la no-op neu dung, va
        # neu label_df CHUA cat thi ma hoa lai rieng o day (ma khong sua luon
        # make_onehot ben duoi) se lam hai ham tra ve key khac nhau -> KeyError.
        for i in range(0, len(columns), 2):
            col_1 = label_df[columns[i]].values
            col_2 = label_df[columns[i + 1]].values
            hla_name = columns[i].split("_")[0]
            combined = sorted({value for value in np.concatenate((col_1, col_2))
                               if _valid_allele(value)})
            if not combined:
                raise ValueError(f"No valid training HLA alleles for {hla_name}")
            one_hot = np.eye(len(combined))
            for j, value in enumerate(combined):
                self.encoder[(hla_name, value)] = one_hot[j]
                self.decoder[(hla_name, j)] = value
            self.label_counter[hla_name] = len(combined)

    def make_onehot(self, label_df, columns):
        onehot = {}
        for col in columns:
            gene = col.split("_")[0]
            zero = np.zeros(self.label_counter[gene])
            onehot[col] = label_df[col].apply(
                lambda x, gene=gene, zero=zero: self.encoder.get((gene, x), zero))
        onehot = pd.DataFrame(onehot, index=label_df.index)
        left = onehot[columns[::2]].rename(index=lambda x: x + "_1",
                                           columns=lambda x: x.replace("_1", ""))
        right = onehot[columns[1::2]].rename(index=lambda x: x + "_2",
                                             columns=lambda x: x.replace("_2", ""))
        return pd.concat([left, right], axis=0)


def load_dataset(vcf_path, labels, markers, group, n_digits, mode, phased,
                 encoder_path=None, keep_samples=None, encoder=None) -> dict:
    """mode in {'train', 'test', 'unlabeled'}. Cung khoa voi AEHLA: data, label,
    input-size, outputs-size, sample-list, columns, encoder, decoder, n_digits.
    'unlabeled' (labels=None) bo khoi nhan va tra 'label' rong -- cong C1 can
    no, va S1 cua HAN cung chay duong nay.

    Vocab fits only samples loaded from the reference/train VCF. Validation
    reuses `encoder=train_dataset['encoder']`; unseen alleles never become
    output classes. `label` is carrier 0/1, `dosage` retains 0/1/2 copies.
    `label-mask` excludes incomplete or unseen genotypes from supervised loss;
    `valid-copies` and `unseen-dosage` retain unseen truth for evaluation.
    """
    if mode not in ("train", "test", "unlabeled"):
        raise ValueError(
            "mode must be train|test|unlabeled, got {!r}".format(mode))
    if mode in ("train", "test") and labels is None:
        raise ValueError("mode={!r} needs labels".format(mode))
    if encoder_path is not None:
        raise NotImplementedError("loading an encoder from file is not ported yet")

    df = load_haplotypes(vcf_path, markers, group, absent_value=-1,
                         require_phased=phased)

    if keep_samples is not None:
        keep = set(map(str, keep_samples))
        df = df[[str(i)[:-2] in keep for i in df.index]]
        if df.empty:
            raise ValueError("keep_samples matches no sample in {}".format(vcf_path))

    # Kiem cap hap NGAY sau khi nap, truoc encoder -- bug 2026-08-28 ben AEHLA
    # gan haplotype cua mau nay cho ten mau khac ma khong bao gi (xem
    # data_helper.py:66-76 / preprocess_data.py:66-76).
    names = sorted(df.index.to_list())
    if len(names) % 2:
        raise ValueError(
            "VCF haplotype rows must come in pairs, got {}".format(len(names)))
    for a, b in zip(names[::2], names[1::2]):
        if not (a.endswith("_1") and b.endswith("_2") and a[:-2] == b[:-2]):
            raise ValueError("haplotype pair misaligned: {} / {}".format(a, b))

    columns = [gene.upper() + "_" + x for gene in GROUPS[group] for x in ("1", "2")]

    encoded = None
    if mode in ("train", "test"):
        df_ids = set(df.index)
        label_ids = {f"{sample}_{hap}" for sample in labels.index for hap in (1, 2)}
        keep_ids = df_ids & label_ids
        dropped = len(df_ids) - len(keep_ids)
        if dropped:
            warnings.warn("Drop {} samples from {} dataset that not in label file"
                         .format(dropped, mode))
        if not keep_ids:
            raise ValueError(
                "0 samples survive the VCF/label-file intersection for mode={!r} "
                "-- check that sample ids in the VCF and the label file actually "
                "match".format(mode))
        df = df[df.index.isin(keep_ids)]
        labels = labels.loc[sorted({str(key)[:-2] for key in df.index})]

    if encoder is None:
        encoder = _Encoder()
        if labels is not None:
            encoder.make(labels, columns)
    if mode in ("train", "test"):
        encoded = encoder.make_onehot(labels, columns)
        encoded["label"] = encoded.apply(lambda x: np.concatenate(x.values), axis=1)

    sample_list = sorted(df.index.to_list())
    dataset_data, dataset_dosage, label_mask = [], [], []
    valid_copies, unseen_dosage = [], []
    sizes = list(encoder.label_counter.values())
    for i in range(0, len(sample_list), 2):
        hap_1 = df.loc[sample_list[i]].values
        hap_2 = df.loc[sample_list[i + 1]].values
        hap_1, hap_2, missing = _split_missing(hap_1, hap_2)
        row_1 = np.logical_or(hap_1, hap_2) * 1
        row_2 = np.logical_and(hap_1, hap_2) * 1
        channels = [row_1, row_2, missing]
        if phased:
            # AE_LPHASE_ORACLE: hang hap1 (da zero-hoa missing) lam dau vao pha,
            # luon la hang CUOI -- xem preprocess_data.py:131-134.
            channels.append(hap_1)
        dataset_data.append(np.stack(channels))
        if encoded is not None:
            dosage = (encoded["label"].loc[sample_list[i]]
                      + encoded["label"].loc[sample_list[i + 1]])
            dataset_dosage.append(dosage)
            valid = np.array([sum(_valid_allele(value) for value in
                                  labels.loc[sample_list[i][:-2], columns[j:j + 2]])
                              for j in range(0, len(columns), 2)])
            known = np.array([block.sum() for block in
                              np.split(dosage, np.cumsum(sizes)[:-1])])
            valid_copies.append(valid)
            unseen_dosage.append(valid - known)
            label_mask.append(np.repeat(known == 2, sizes))

    # mode='unlabeled' replaces AEHLA's load_unlabeled_dataset (the function S1
    # pretraining actually calls) rather than preprocess_data.load_dataset's own
    # mode='unlabeled' passthrough -- match ITS dtype/type, not the labeled
    # path's: float32 instead of int is 46MB instead of 367MB on HAN g1, and
    # Task 2 runs S1 through exactly this path. np.array_equal (the C1 gate) is
    # dtype-blind, so this doesn't touch C1's bit-for-bit result.
    unlabeled = mode == "unlabeled"
    data = np.asarray(dataset_data, dtype=np.float32 if unlabeled else None)
    return {
        "data": data,
        "label": (np.array(dataset_dosage) > 0).astype(int),
        "dosage": np.array(dataset_dosage),
        "label-mask": np.array(label_mask, dtype=bool),
        "valid-copies": np.array(valid_copies),
        "unseen-dosage": np.array(unseen_dosage),
        "input-size": int(data.shape[-1]),
        "outputs-size": [["HLA_" + col, encoder.label_counter[col]]
                        for col in encoder.label_counter],
        "sample-list": np.array([sample_list[i][:-2]
                                 for i in range(0, len(sample_list), 2)]),
        "columns": columns,
        "encoder": encoder,
        "decoder": encoder.decoder,
        "n_digits": n_digits,
        "type": "unlabeled-collapsed" if unlabeled else "collapsed",
    }

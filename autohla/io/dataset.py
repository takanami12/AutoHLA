"""Nap VCF + nhan thanh tensor huan luyen.

Port cua ban goc preprocess_data.load_dataset (dong 59-223), rut gon: bo cac
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


def _split_missing(hap_1, hap_2):
    """(-1 sentinel tu load_haplotypes) -> haplotype sach + kenh missing."""
    missing = ((hap_1 < 0) | (hap_2 < 0)) * 1
    return np.where(missing, 0, hap_1), np.where(missing, 0, hap_2), missing


class _Encoder:
    """Ban rut gon cua ban goc objects.encoder.Encoder -- chi giu phan
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
            combined = sorted(set(np.concatenate((col_1, col_2), axis=0)))
            one_hot = np.eye(len(combined))
            for j, value in enumerate(combined):
                self.encoder[(hla_name, value)] = one_hot[j]
                self.decoder[(hla_name, j)] = value
            self.label_counter[hla_name] = len(combined)

    def make_onehot(self, label_df, columns):
        onehot = {col: label_df[col].apply(lambda x: self.encoder[(col.split("_")[0], x)])
                 for col in columns}
        onehot = pd.DataFrame(onehot, index=label_df.index)
        left = onehot[columns[::2]].rename(index=lambda x: x + "_1",
                                           columns=lambda x: x.replace("_1", ""))
        right = onehot[columns[1::2]].rename(index=lambda x: x + "_2",
                                             columns=lambda x: x.replace("_2", ""))
        return pd.concat([left, right], axis=0)


def _collapse_label(encoded, key_1, key_2):
    """(nhan multi-hot 0/1 cho BCE, lieu THAT 0/1/2 -- 2 khi dong hop).

    OR huy mat ban sao thu hai cua dong hop, ma do dung la thu `tune_tau`/
    `fit_beta` can do (tau CHINH LA boi so dong hop/di hop). Nen tra ca hai:
    nhan BCE giu nguyen bit-for-bit, lieu that lay tu tong.
    """
    a, b = encoded["label"].loc[key_1], encoded["label"].loc[key_2]
    return np.logical_or(a, b) * 1, a + b


def load_dataset(vcf_path, labels, markers, group, n_digits, mode, phased,
                 encoder_path=None, keep_samples=None) -> dict:
    """mode in {'train', 'test', 'unlabeled'}. Cung khoa voi ban goc: data, label,
    input-size, outputs-size, sample-list, columns, encoder, decoder, n_digits.
    'unlabeled' (labels=None) bo khoi nhan va tra 'label' rong -- cong C1 can
    no, va S1 cua HAN cung chay duong nay.

    `keep_samples` gioi han MAU duoc nap ma KHONG dong cham vao tu vung allele:
    encoder van xay tu `labels` day du. Hai chuyen do khac nhau -- cat split noi
    bo tu mot VCF duy nhat (cli.py) can chon mau, nhung tu vung phai giu nguyen,
    khong thi train va val danh so allele khac nhau. None = nap het (mac dinh,
    duong ma cac cong C1/C3 di qua).
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

    # Kiem cap hap NGAY sau khi nap, truoc encoder -- bug 2026-08-28 ben ban goc
    # gan haplotype cua mau nay cho ten mau khac ma khong bao gi (xem
    # data_helper / preprocess_data).
    names = sorted(df.index.to_list())
    if len(names) % 2:
        raise ValueError(
            "VCF haplotype rows must come in pairs, got {}".format(len(names)))
    for a, b in zip(names[::2], names[1::2]):
        if not (a.endswith("_1") and b.endswith("_2") and a[:-2] == b[:-2]):
            raise ValueError("haplotype pair misaligned: {} / {}".format(a, b))

    columns = [gene.upper() + "_" + x for gene in GROUPS[group] for x in ("1", "2")]

    encoder = _Encoder()
    if labels is not None:
        encoder.make(labels, columns)

    encoded = None
    if mode in ("train", "test"):
        encoded = encoder.make_onehot(labels, columns)
        df_ids, label_ids = set(df.index), set(encoded.index)
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
        encoded = encoded[encoded.index.isin(keep_ids)]
        df = df[df.index.isin(keep_ids)]
        encoded["label"] = encoded.apply(lambda x: np.concatenate(x.values), axis=1)

    sample_list = sorted(df.index.to_list())
    dataset_data, dataset_label, dataset_dosage = [], [], []
    for i in range(0, len(sample_list), 2):
        hap_1 = df.loc[sample_list[i]].values
        hap_2 = df.loc[sample_list[i + 1]].values
        hap_1, hap_2, missing = _split_missing(hap_1, hap_2)
        row_1 = np.logical_or(hap_1, hap_2) * 1
        row_2 = np.logical_and(hap_1, hap_2) * 1
        channels = [row_1, row_2, missing]
        if phased:
            # AE_LPHASE_ORACLE: hang hap1 (da zero-hoa missing) lam dau vao pha,
            # luon la hang CUOI -- xem preprocess_data.
            channels.append(hap_1)
        dataset_data.append(np.stack(channels))
        if encoded is not None:
            label, dose = _collapse_label(encoded, sample_list[i],
                                          sample_list[i + 1])
            dataset_label.append(label)
            dataset_dosage.append(dose)

    # mode='unlabeled' replaces ban goc's load_unlabeled_dataset (the function S1
    # pretraining actually calls) rather than preprocess_data.load_dataset's own
    # mode='unlabeled' passthrough -- match ITS dtype/type, not the labeled
    # path's: float32 instead of int is 46MB instead of 367MB on HAN g1, and
    # Task 2 runs S1 through exactly this path. np.array_equal (the C1 gate) is
    # dtype-blind, so this doesn't touch C1's bit-for-bit result.
    unlabeled = mode == "unlabeled"
    data = np.asarray(dataset_data, dtype=np.float32 if unlabeled else None)
    return {
        "data": data,
        "label": np.array(dataset_label),
        "dosage": np.array(dataset_dosage, dtype=float),
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

#!/usr/bin/env python3
"""Dung du lieu cho hai thi nghiem external test (train FULL cohort -> test 1KGP).

    vn1k_khv: train VN1K 996 mau (GRCh38)  -> test KHV 99 mau
    han_chb: train HAN 10.488 mau (hg18)  -> test CHB 103 mau

Luoi marker = APMDA giao voi CA HAI VCF (train va test), nen khong ai duoc nhin
marker ma phia kia khong co. Ket qua duoc do vao DUNG cay thu muc cua giao thuc
CV san co duoi so fold 91, de moi runner (SNP2HLA/CookHLA/HIBAG/DEEP*HLA/AutoHLA)
chay nguyen ven khong phai viet lai.

    python prepare [vn1k_khv|han_chb...]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from common import LOCI, TARGET_COLS, normalize_allele  # noqa: E402

FOLD = 91
G1K_LABEL = ROOT / "data/label/G1K_HLA-LA_Ggroup.resolution.4digits.csv"

CONFIGS = {
    "vn1k_khv": dict(
        train_vcf=ROOT / "data/VN1K/VN1K.MHC.vcf.gz",
        test_vcf=ROOT / "data/external_test/by_reference/VN1K_GRCh38/KHV.MHC.vcf.gz",
        train_labels=ROOT / "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv",
        chip=ROOT / "data/references/ref.APMDA.position.list",
        markers=ROOT / "data/references/ref.APMDA.KHV.position.list",
        chrom="chr6",
        data=ROOT / "results/cv_vn1k/data" / f"fold{FOLD:02d}",
        proto=ROOT / "results/cv_vn1k/protocol" / f"fold{FOLD:02d}",
        label_out=ROOT / "data/label/ext_VN1K_KHV.csv",
        sep=",",
    ),
    "han_chb": dict(
        train_vcf=ROOT / "HAN_dataset/HAN.hg18.vcf.gz",
        test_vcf=ROOT / "data/external_test/by_reference/HAN_hg18/CHB.MHC.vcf.gz",
        train_labels=ROOT / "HAN_dataset/HAN.HLA.4digit.tsv",
        chip=ROOT / "data/references/ref.APMDA.hg18.position.list",
        markers=ROOT / "data/references/ref.APMDA_CHB.hg18.position.list",
        chrom="6",
        data=ROOT / "results/cv_han/data/APMDA_CHB" / f"fold{FOLD:02d}",
        proto=ROOT / "results/cv_han/protocol" / f"fold{FOLD:02d}",
        label_out=ROOT / "data/label/ext_HAN_CHB.tsv",
        sep="\t",
    ),
}


def sh(*cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True, **kw)


def vcf_keys(vcf):
    """({(pos, REF, ALT)}, {pos: so ban ghi}) cua mot VCF."""
    out = subprocess.run(["bcftools", "query", "-f", "%POS\t%REF\t%ALT\n", str(vcf)],
                         capture_output=True, text=True, check=True).stdout
    keys, per_pos = set(), {}
    for line in out.splitlines():
        key = tuple(line.split("\t"))
        keys.add(key)
        per_pos[key[0]] = per_pos.get(key[0], 0) + 1
    return keys, per_pos


def chip_keys(path):
    keys = []
    for line in path.read_text().splitlines():
        f = line.split()
        keys.append((f[1], f[2].upper(), f[3].upper()))
    return keys


def write_markers(cfg):
    """Luoi chung = chip APMDA ∩ VCF train ∩ VCF test, giu thu tu theo toa do."""
    train_keys, train_pos = vcf_keys(cfg["train_vcf"])
    test_keys, test_pos = vcf_keys(cfg["test_vcf"])
    shared = train_keys & test_keys
    # Vi tri da allele bi loai han: bcftools -T khop theo TOA DO, nen giu lai thi
    # train va test se lech so hang va khong con cung mot luoi marker.
    keys = sorted({k for k in chip_keys(cfg["chip"]) if k in shared
                   and train_pos[k[0]] == 1 and test_pos[k[0]] == 1},
                  key=lambda k: int(k[0]))
    if not keys:
        raise SystemExit(f"khong co marker chung: {cfg['chip']}")
    cfg["markers"].write_text(
        "".join(f"{cfg['chrom']}\t{p}\t{r}\t{a}\n" for p, r, a in keys))
    print(f"{cfg['markers'].name}: {len(keys)} marker")
    return keys


def load_train_labels(cfg):
    path = cfg["train_labels"]
    sep = "\t" if path.suffix == ".tsv" else ","
    raw = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    raw = raw.rename(columns={raw.columns[0]: "sample_id"}).set_index("sample_id")
    out = pd.DataFrame("", index=raw.index, columns=TARGET_COLS, dtype=object)
    for col in TARGET_COLS:
        if col in raw.columns:
            out[col] = raw[col].map(
                lambda v: v.split("*", 1)[1] if isinstance(v, str) and "*" in v else v)
    return out


def load_g1k_labels(ids):
    raw = pd.read_csv(G1K_LABEL, dtype=str).fillna("")
    raw = raw.rename(columns={raw.columns[0]: "sample_id"}).set_index("sample_id")
    missing = [s for s in ids if s not in raw.index]
    if missing:
        raise SystemExit(f"thieu nhan 1KGP cho {len(missing)} mau: {missing[:5]}")
    return raw.reindex(index=ids, columns=TARGET_COLS).fillna("")


def samples_of(vcf):
    return subprocess.run(["bcftools", "query", "-l", str(vcf)],
                          capture_output=True, text=True, check=True).stdout.split()


def subset(vcf, samples_file, targets, out):
    if out.exists() and out.stat().st_size:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    sh("bcftools", "view", "-S", samples_file, "-T", targets, "-Oz", "-o", out, vcf)
    sh("tabix", "-f", "-p", "vcf", out)


def build(name):
    cfg = CONFIGS[name]
    cfg["proto"].mkdir(parents=True, exist_ok=True)
    cfg["data"].mkdir(parents=True, exist_ok=True)
    keys = write_markers(cfg)
    targets = cfg["data"] / "markers.targets"
    targets.write_text("".join(f"{cfg['chrom']}\t{p}\t{r}\t{a}\n" for p, r, a in keys))

    train_labels = load_train_labels(cfg)
    pool = [s for s in samples_of(cfg["train_vcf"]) if s in train_labels.index]
    # 5% validation lay ra tu chinh tap train (quy tac giao thuc), seed 77.
    val = sorted(pd.Series(pool).sample(frac=0.05, random_state=77))
    train = [s for s in pool if s not in set(val)]
    test = samples_of(cfg["test_vcf"])
    if set(test) & set(pool):
        raise SystemExit("mau test nam trong cohort train")
    test_labels = load_g1k_labels(test)

    for split, ids in (("train", train), ("val", val), ("test", test)):
        (cfg["proto"] / f"{split}.samples").write_text("\n".join(ids) + "\n")
        frame = (test_labels if split == "test" else train_labels).reindex(ids)
        frame.index.name = "sample_id"
        frame.to_csv(cfg["proto"] / f"{split}.labels.tsv", sep="\t")
        frame.to_csv(cfg["data"] / f"{split}.labels.tsv", sep="\t")
        src = cfg["test_vcf"] if split == "test" else cfg["train_vcf"]
        subset(src, cfg["proto"] / f"{split}.samples", targets,
               cfg["data"] / f"{split}.vcf.gz")

    # Bang nhan GOP: evaluator lay allele universe + AF tu train va truth tu test,
    # nen ca hai phai nam trong MOT file.
    merged = pd.concat([train_labels.reindex(pool), test_labels])
    merged = merged.map(lambda v: normalize_allele(v) or "")
    merged.index.name = "sample_id"
    merged.to_csv(cfg["label_out"], sep=cfg["sep"])

    counts = {s: len(bcf_samples) for s, bcf_samples in
              (("train", train), ("val", val), ("test", test))}
    print(f"{name}: {counts}, {len(keys)} marker, nhan -> {cfg['label_out'].name}")
    for split in ("train", "val", "test"):
        n = len(subprocess.run(
            ["bcftools", "view", "-H", str(cfg["data"] / f"{split}.vcf.gz")],
            capture_output=True, text=True, check=True).stdout.splitlines())
        assert n == len(keys), f"{split}.vcf.gz co {n} marker, cho {len(keys)}"
    universe = {g: sorted({v for c in (1, 2)
                           for v in merged[f"{g}_{c}"] if v}) for g in LOCI}
    print("  allele universe:", {g: len(v) for g, v in universe.items()})


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CONFIGS)):
        build(name)

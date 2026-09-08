"""Shared helpers for converting method outputs to the target CSV layout.

Target CSV layout (matches do noi bo<ds>/hla_imputation/<fold>/*.vcf.csv):
  - first header cell empty, then A_1,A_2,B_1,B_2,C_1,C_2,DPB1_1,DPB1_2,
    DRB1_1,DRB1_2,DQA1_1,DQA1_2,DQB1_1,DQB1_2
  - row index = sample ID
  - alleles are 2-field, e.g. "11:01"
"""
from __future__ import annotations

import csv
import functools
import os

import pandas as pd

LOCI = ["A", "B", "C", "DPB1", "DRB1", "DQA1", "DQB1"]
TARGET_COLS = [f"{loc}_{i}" for loc in LOCI for i in (1, 2)]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASETS = ["VN1K", "1KGP_VN1K"]
FOLDS = ["B301", "B303", "B304", "B305"]

# Ground-truth label file per dataset (both share the 14-column layout, but the
# locus order differs from the target; we always key by column name).
LABEL_FILES = {
    "VN1K": os.path.join(REPO, "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv"),
    "1KGP_VN1K": os.path.join(REPO, "data/label/1KGP_VN1K.HLA.resolution.4digits.csv"),
    "HAN": os.path.join(REPO, "HAN_dataset/HAN.HLA.4digit.tsv"),
    "VN1K_KHV": os.path.join(REPO, "data/label/ext_VN1K_KHV.ggroup.csv"),
    "HAN_CHB": os.path.join(REPO, "data/label/ext_HAN_CHB.ggroup.tsv"),
}

_NA_TOKENS = {"", "NA", "na", "N/A", "-", ".", "nan", "None"}


def normalize_allele(value) -> str | None:
    """Return a clean 2-field allele string, or None for a no-call."""
    if value is None:
        return None
    s = str(value).strip().strip('"')
    if s in _NA_TOKENS:
        return None
    if ":" in s:
        fam, _, prot = s.partition(":")
        fam = fam.lstrip("0") or "0"
        if len(fam) < 2:
            fam = fam.zfill(2)
        if len(prot) < 2:
            prot = prot.zfill(2)
        return f"{fam}:{prot}"
    return s  # numeric code without colon; caller should use code_to_2field


def _fmt(fam: str, prot: str) -> str:
    f = fam.lstrip("0") or "0"
    if len(f) < 2:
        f = f.zfill(2)
    if len(prot) < 2:
        prot = prot.zfill(2)
    return f"{f}:{prot}"


def code_to_2field(code, known: set | None = None) -> str | None:
    """Convert a colon-less numeric allele code (e.g. '1101', '10401', '104')
    to a 2-field allele. Ambiguous codes (3-digit family or 3-digit protein) are
    disambiguated against `known` (the set of valid alleles for the locus, taken
    from the label). Falls back to the standard last-2-digits-are-protein split.
    """
    if code is None:
        return None
    s = str(code).strip().strip('"')
    if s in _NA_TOKENS:
        return None
    if ":" in s:
        return normalize_allele(s)
    if not s.isdigit():
        return None

    candidates: list[str] = []
    if len(s) >= 3:  # last 2 digits = protein  (1101 -> 11:01, 10401 -> 104:01)
        candidates.append(_fmt(s[:-2], s[-2:]))
    if len(s) >= 4:  # last 3 digits = protein  (06130 -> 06:130, 14141 -> 14:141)
        candidates.append(_fmt(s[:-3], s[-3:]))
    candidates.append(_fmt(s, "01"))  # family-only (104 -> 104:01)

    if known:
        for c in candidates:
            if c in known:
                return c
    return candidates[0]


def load_label(dataset: str) -> pd.DataFrame:
    """Return label as DataFrame indexed by sample, columns = TARGET_COLS."""
    path = LABEL_FILES[dataset]
    # HAN dung tab, VN1K dung dau phay -- do dau phan cach thay vi cung hoa.
    with open(path) as fh:
        sep = "\t" if "\t" in fh.readline() else ","
    df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
    df = df.rename(columns={df.columns[0]: "sample"}).set_index("sample")
    out = pd.DataFrame(index=df.index, columns=TARGET_COLS, dtype=object)
    for col in TARGET_COLS:
        # Nhan HAN ghi "A*33:03"; du doan chi mang "33:03". Bo tien to gene.
        values = (df[col].map(lambda v: v.split("*", 1)[1] if isinstance(v, str)
                              and "*" in v else v)
                  if col in df.columns else None)
        out[col] = values.map(normalize_allele) if values is not None else None
    return out


def known_alleles(dataset: str) -> dict[str, set]:
    """Per-locus set of valid 2-field alleles seen in the label (used to
    disambiguate numeric codes)."""
    lab = load_label(dataset)
    out: dict[str, set] = {}
    for loc in LOCI:
        vals = set()
        for i in (1, 2):
            vals |= {v for v in lab[f"{loc}_{i}"].dropna().unique() if v}
        out[loc] = vals
    return out


# ------------------------------------------------------------------ G-group ---
# Bang G-group cua IPD-IMGT/HLA (wmda/hla_nom_g.txt). Hai bo goi allele dat ten
# KHAC nhau cho cung mot G-group -- nhan VN1K la HISAT (`DRB1*14:54`), nhan KHV la
# HLA-LA (`DRB1*14:01`) -- nen truth va prediction lech nhau ve TEN chu khong phai
# ve sinh hoc. Gop ve ten dai dien khoa lech do lai, DONG NHAT cho moi method.
GGROUP_FILE = os.path.join(REPO, "data/references/hla_nom_g.txt")
GGROUP = os.environ.get("HLA_GGROUP", "0").lower() not in {"", "0", "off", "false"}


def two_field(value) -> str | None:
    """Cat allele bao nhieu field cung duoc ve dung 2 field ('01:01:01:02N' -> '01:01')."""
    allele = normalize_allele(value)
    if allele is None or ":" not in allele:
        return allele
    fam, _, rest = allele.partition(":")
    return _fmt(fam, rest.split(":")[0])


@functools.lru_cache(maxsize=1)
def ggroup_map() -> dict:
    """{(gene, allele 2-field): ten dai dien 2-field cua G-group}.

    Chi gop khi mot ten 2-field nam tron trong DUNG mot G-group. Trai tren nhieu
    G-group (DQA1*03:03, C*02:02...) thi khong co trong map -- giu nguyen ten,
    vi gop them se xoa mot phan biet co that.
    """
    seen: dict[tuple[str, str], set] = {}
    with open(GGROUP_FILE) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(";")
            if len(parts) < 3 or not parts[2]:
                continue
            gene = parts[0].rstrip("*")
            rep = two_field(parts[2].rstrip("G"))
            for allele in parts[1].split("/"):
                seen.setdefault((gene, two_field(allele)), set()).add(rep)
    return {key: reps.pop() for key, reps in seen.items() if len(reps) == 1}


def empty_target_frame(samples) -> pd.DataFrame:
    df = pd.DataFrame(index=list(samples), columns=TARGET_COLS, dtype=object)
    df.index.name = None
    return df


def write_target_csv(df: pd.DataFrame, path: str) -> None:
    """Write in the exact target layout: empty first header cell, sample index."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.reindex(columns=TARGET_COLS)
    df.index.name = None
    df.to_csv(path, quoting=csv.QUOTE_MINIMAL)

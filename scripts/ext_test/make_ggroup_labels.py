#!/usr/bin/env python3
"""Hoa hop quy uoc TEN allele giua hai bo goi, ghi ra mot FILE NHAN MOI.

O external test co hai dau nhan do hai bo goi khac nhau sinh ra: train la
HISAT-genotype (tra ten allele cu the: `C*07:06`, `DRB1*14:54`), test la
HLA-LA da o dang **G-group** (`C*07:01`, `DRB1*14:01`). Cung mot haplotype, hai
cai ten. Khong hoa hop thi moi method deu bi phat vi mot khac biet ve QUY UOC,
khong phai ve sinh hoc (do duoc: top-2 recall bin `5-10%` 0,829 so voi 0,980 sau
khi gop -- `do noi bo` muc 2).

Cong cu nay dung mot minh, KHONG phai mot buoc trong `prepare.py` va KHONG phai
mot co trong bat ky mo hinh nao. Vao mot file nhan, ra mot file nhan. Chuoi la:

    prepare.py  ->  make_ggroup_labels.py  ->  moi method

Ban do lay tu IPD-IMGT/HLA `data/references/hla_nom_g.txt` qua
`common.ggroup_map`: chi gop khi mot ten 2-field nam TRON trong dung mot
G-group; ten trai tren nhieu G-group (`DQA1*03:03`, `C*02:02`...) giu nguyen,
vi gop them se xoa mot phan biet co that.

    python make_ggroup_labels            # ca hai o
    python make_ggroup_labels vn1k_khv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from common import LOCI, TARGET_COLS, ggroup_map, normalize_allele  # noqa: E402

CELLS = {
    "vn1k_khv": dict(
        source=ROOT / "data/label/ext_VN1K_KHV.csv",
        out=ROOT / "data/label/ext_VN1K_KHV.ggroup.csv",
        proto=ROOT / "results/cv_vn1k/protocol/fold91",
        sep=","),
    "han_chb": dict(
        source=ROOT / "data/label/ext_HAN_CHB.tsv",
        out=ROOT / "data/label/ext_HAN_CHB.ggroup.tsv",
        proto=ROOT / "results/cv_han/protocol/fold91",
        sep="\t"),
}


def collapse_cell(value, gene, gmap):
    """Giu NGUYEN hinh dang van ban cua o: tien to `GENE*` neu co, rong neu rong.

    Doi hinh dang o day se lam cac reader phia sau (make_hla_ped.py, HIBAG fmt,
    common.load_label) doc ra thu khac -- moi cai chiu duoc ca hai dang, nhung
    chung khong nhat thiet doc GIONG nhau, nen file moi phai khac file cu dung o
    phan noi dung."""
    if not isinstance(value, str) or not value.strip():
        return value
    raw = value.strip()
    prefix, _, rest = raw.rpartition("*")
    allele = normalize_allele(rest)
    if allele is None:
        return value
    mapped = gmap.get((gene, allele), allele)
    return f"{prefix}*{mapped}" if prefix else mapped


def build(name):
    cfg = CELLS[name]
    gmap = ggroup_map()
    frame = pd.read_csv(cfg["source"], sep=cfg["sep"], dtype=str).fillna("")
    first = frame.columns[0]
    out = frame.copy()

    train_ids = set((cfg["proto"] / "train.samples").read_text().split())
    val_ids = set((cfg["proto"] / "val.samples").read_text().split())
    test_ids = set((cfg["proto"] / "test.samples").read_text().split())
    side = {**{s: "train" for s in train_ids | val_ids},
            **{s: "test" for s in test_ids}}

    renames, totals = [], {"train": 0, "test": 0, "khac": 0}
    for column in TARGET_COLS:
        if column not in frame.columns:
            continue
        gene = column.split("_")[0]
        out[column] = [collapse_cell(v, gene, gmap) for v in frame[column]]
        for sid, before, after in zip(frame[first], frame[column], out[column]):
            if before != after:
                where = side.get(str(sid), "khac")
                totals[where] += 1
                renames.append(dict(gene=gene, tu=before, thanh=after, phia=where))

    # Gop phai la LUY DANG: ten dai dien cua mot G-group phai anh xa ve chinh no.
    # Neu sai, chay tool hai lan ra hai file khac nhau va khong ai biet.
    twice = {c: [collapse_cell(v, c.split("_")[0], gmap) for v in out[c]]
             for c in TARGET_COLS if c in out.columns}
    for column, values in twice.items():
        if list(out[column]) != values:
            raise SystemExit(f"gop khong luy dang o cot {column} -- kiem hla_nom_g.txt")

    out.to_csv(cfg["out"], sep=cfg["sep"], index=False)

    def universe(f):
        return {g: len({normalize_allele(v.rpartition("*")[2])
                        for c in (1, 2) for v in f[f"{g}_{c}"] if str(v).strip()}
                       - {None}) for g in LOCI}

    before_u, after_u = universe(frame), universe(out)
    table = (pd.DataFrame(renames)
             .groupby(["gene", "tu", "thanh", "phia"], as_index=False)
             .size().rename(columns={"size": "ban_sao"})
             .sort_values("ban_sao", ascending=False))
    audit = ROOT / f"data/label/ggroup_rename_{name}.csv"
    table.to_csv(audit, index=False)

    n_cells = sum(len(frame) for c in TARGET_COLS if c in frame.columns)
    print(f"[{name}] {cfg['source'].name} -> {cfg['out'].name}")
    print(f"  ban sao doi ten: train {totals['train']}, test {totals['test']}"
          f", ngoai protocol {totals['khac']}  /  {n_cells} o")
    print(f"  universe: {sum(before_u.values())} -> {sum(after_u.values())} allele"
          f"  ({ {g: (before_u[g], after_u[g]) for g in LOCI} })")
    print(f"  bang doi ten -> {audit}")
    if not table.empty:
        print(table.head(8).to_string(index=False))
    return table


if __name__ == "__main__":
    for cell in (sys.argv[1:] or list(CELLS)):
        build(cell)
        print()

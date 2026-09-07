#!/usr/bin/env python3
"""Harness CV = caller MONG cua `autohla.cli.main`.

Cai duong nghien cuu cu (scripts/cv_vn1k/*.sh + mot dong bien moi truong
PAIR_*/BLEND_*/AE_*) chay mot duong ma nguoi dung ngoai KHONG chay duoc. O day
moi o goi dung hai lenh cong khai: `train` roi `impute`. Neu harness khong tai
lap duoc so, do la loi cua goi chu khong phai cua mot lop vá rieng.

    python tools/run_cv.py --cohort vn1k --folds 1-10 --groups 1-4 \
        --out results/autohla_cv_vn1k --par 32 --threads 4

Idempotent: o nao da co calls.csv thi bo qua (40 o la nhieu gio).
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "vn1k": dict(data=ROOT / "results/cv_vn1k/data",
                 labels=ROOT / "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv",
                 pos=ROOT / "data/references/ref.APMDA.position.list",
                 phase="off"),
    # VN1K da pha TRONG fold (panel = train+val cua chinh fold, khong co marker
    # HLA). Cung mau, cung protocol, cung danh sach marker -- chi khac file VCF.
    "vn1k_phased": dict(data=ROOT / "results/lphase_infold/data",
                        labels=ROOT / "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv",
                        pos=ROOT / "data/references/ref.APMDA.position.list",
                        phase="on"),
    # External test: train FULL cohort, test tren 1KGP. Cung cay thu muc CV,
    # fold 91, luoi marker = APMDA giao voi ca hai VCF.
    "vn1k_khv": dict(data=ROOT / "results/cv_vn1k/data",
                     labels=ROOT / "data/label/ext_VN1K_KHV.ggroup.csv",
                     pos=ROOT / "data/references/ref.APMDA.KHV.position.list",
                     phase="off"),
    # VN1K->KHV nhung ca hai dau DA PHA TRONG CHINH O DO: panel = train+val cua
    # fold 91 (996 mau VN1K), chi marker chip, khong marker HLA; KHV duoc pha bang
    # panel do. Cung thu tuc `rephase_infold.sh` da dung cho 10 fold noi bo, nen
    # khong co duong ro ri va lam duoc tren du lieu moi.
    "vn1k_khv_phased": dict(data=ROOT / "results/lphase_infold/data",
                            labels=ROOT / "data/label/ext_VN1K_KHV.ggroup.csv",
                            pos=ROOT / "data/references/ref.APMDA.KHV.position.list",
                            phase="on"),
    "han_chb": dict(data=ROOT / "results/cv_han/data/APMDA_CHB",
                    labels=ROOT / "data/label/ext_HAN_CHB.ggroup.tsv",
                    pos=ROOT / "data/references/ref.APMDA_CHB.hg18.position.list",
                    phase="off", build="hg18"),
    "han": dict(data=ROOT / "results/cv_han/data/APMDA",
                labels=ROOT / "HAN_dataset/HAN.HLA.4digit.tsv",
                pos=ROOT / "data/references/ref.APMDA.hg18.position.list",
                phase="off", build="hg18"),
}


@dataclass(frozen=True)
class Cell:
    cohort: str
    fold: int
    group: int
    head: str = "full"

    @property
    def name(self):
        return f"fold{self.fold:02d}_g{self.group}"

    @property
    def arm(self):
        return "base" if self.head == "full" else "base_leanhead"

    def done(self, out_root):
        return (cell_dir(out_root, self) / "calls.csv").exists()


def cell_dir(out_root, cell: Cell) -> Path:
    # `run_cell` gọi module từ repository root nên --out giữ nguyên vị trí.
    return Path(out_root).resolve() / cell.arm / cell.name


def plan_cells(folds, groups, cohort="vn1k", head="full"):
    return [Cell(cohort, int(f), int(g), head)
            for f in folds for g in groups]


def _parse_range(text):
    if "-" in text:
        lo, hi = text.split("-", 1)
        return range(int(lo), int(hi) + 1)
    return [int(x) for x in text.split(",")]


def run_cell(cell: Cell, out_root, epochs, s1_epochs, threads, posthoc_folds,
             marker_list=True, pair="on"):
    """Hai lenh cong khai, khong gi khac. Tra (cell, ok, log)."""
    spec = COHORTS[cell.cohort]
    fold = spec["data"] / f"fold{cell.fold:02d}"
    out = cell_dir(out_root, cell)
    out.mkdir(parents=True, exist_ok=True)
    model = out / "model"
    common = [sys.executable, "-m", "autohla"]
    train = common + [
        "train", "--vcf", str(fold / "train.vcf.gz"), "--labels", str(spec["labels"]),
        "--group", str(cell.group), "--out", str(model),
        "--epochs", str(epochs), "--s1-epochs", str(s1_epochs),
        "--threads", str(threads), "--phase", spec.get("phase", "off"),
        "--head", cell.head,
    ]
    # posthoc_folds=0 -> bo tang ridge+tron. Tang do train THEM k curriculum moi
    # o (k=10 mac dinh la 11 luot/o thay vi 1), va no nam TREN trunk nen khong
    # dinh gi toi trunk.
    train += ["--no-posthoc"] if not posthoc_folds else [
        "--posthoc-folds", str(posthoc_folds), "--pair", pair]
    if marker_list:
        train += ["--marker-list", str(spec["pos"])]
    impute = common + ["impute", "--model", str(model), "--vcf",
                       str(fold / "test.vcf.gz"), "--out", str(out / "calls.csv"),
                       "--threads", str(threads)]
    log = out / "run.log"
    with open(log, "w") as fh:
        for command in (train, impute):
            proc = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(ROOT))
            if proc.returncode != 0:
                return cell, False, str(log)
    return cell, True, str(log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="vn1k", choices=sorted(COHORTS))
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--groups", default="1-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--s1-epochs", type=int, default=100)
    ap.add_argument("--posthoc-folds", type=int, default=0,
                    help="0 = --no-posthoc (mac dinh o day: k fold nhan chi phi len k+1 lan)")
    ap.add_argument("--head", default="full", choices=("full", "lean"))
    ap.add_argument("--pair", default="on", choices=("auto", "on", "off"),
                    help="tang pair; bo qua khi --posthoc-folds 0 (khong co hau ky)")
    args = ap.parse_args(argv)

    cells = plan_cells(_parse_range(args.folds), _parse_range(args.groups),
                       cohort=args.cohort, head=args.head)
    todo = [c for c in cells if not c.done(args.out)]
    print(f"{len(cells)} o, {len(cells) - len(todo)} da xong, chay {len(todo)} "
          f"(par={args.par}, threads={args.threads})", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=args.par) as pool:
        futures = [pool.submit(run_cell, c, args.out, args.epochs, args.s1_epochs,
                               args.threads, args.posthoc_folds, pair=args.pair)
                   for c in todo]
        for future in futures:
            cell, ok, log = future.result()
            print(f"{'OK  ' if ok else 'LOI '} {cell.arm} {cell.name}  {log}",
                  flush=True)
            if not ok:
                failed.append(cell)
    print(f"xong: {len(todo) - len(failed)}/{len(todo)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

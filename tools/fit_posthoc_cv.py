#!/usr/bin/env python3
"""Tang pair + ridge fit HAU KY tren chinh cau truc 10-fold cua sweep.

Vi sao khong dung `--posthoc-folds` cua `autohla train`: vong lap do cat them k
fold NOI BO va huan luyen k+1 curriculum MOI O -- vi duong public chi co mot VCF
va khong san fold nao. O day thi da co 10-fold CV roi: mo hinh cua fold f khong
he nhin thay mau test cua fold f, nen 40 o CV TU CHUNG da la du doan out-of-fold,
dung thu ma beta can. Chi phi vi the la ~1x sweep nen thay vi 11x.

Ba giai doan:
  1. Moi o: nap model da train, (tuy chon) fine-tune tang pair tren train cua
     fold do voi dung som tren val, va fit ridge tren z. Xuat dosage MEM.
  2. Gop 10 fold thanh hai file theo dung schema cua predictions/.
  3. Goi blend_cv_dir -- chinh ham da tai lap beta_*.csv sai so 0 -- de chon beta
     LOFO va xuat goi cung.

    python tools/fit_posthoc_cv.py --sweep results/autohla_cv_vn1k --arm cm_off_nostem \
        --cohort vn1k --name AUTOHLA_NOSTEM
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from run_cv import COHORTS  # noqa: E402

from autohla.io.dataset import load_dataset  # noqa: E402
from autohla.io.labels import load_labels  # noqa: E402
from autohla.io.model_dir import load_model  # noqa: E402
from autohla.io.vcf import GROUPS, read_markers  # noqa: E402
from autohla.model.pair import PairEnergyHead  # noqa: E402
from autohla.model.ridge import apply_ridge, ridge_prob, select_lambda, zscore  # noqa: E402
from autohla.train.pair import predict_pair, train_pair  # noqa: E402

PRED = ROOT / os.environ.get("POSTHOC_PRED_DIR", "results/cv_vn1k/predictions")


def _blocks(outputs_size):
    out, start = [], 0
    for name, size in outputs_size:
        out.append((name.replace("HLA_", ""), slice(start, start + size), size))
        start += size
    return out


def _norm(matrix):
    """Diem ridge -> phan bo tren moi hang. Mot dong bao boc `ridge_prob`.

    Truoc day day la ban sao rieng cua phep dich-theo-min; goi thang ham cua goi
    de duong nghien cuu va duong san pham khong con troi khoi nhau (bug 09-04:
    cli.py van kep bang to_prob trong khi file nay da duoc vá tu 3da8ca02).
    """
    return ridge_prob(matrix)


def _rows(dosage, samples, fold, blocks, decoder, sink):
    for gene, sl, size in blocks:
        block = dosage[:, sl]
        for i, sample in enumerate(samples):
            for a in np.nonzero(block[i] > 1e-6)[0]:
                sink.append((str(sample), fold, gene, decoder[(gene, int(a))],
                             float(block[i, a])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    ap.add_argument("--name", required=True, help="tien to file trong predictions/")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--groups", default="1-4")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pair-epochs", type=int, default=40)
    ap.add_argument("--only", default="both", choices=("both", "ridge", "pair"),
                    help="ridge la dang dong kin nen tinh lai rieng duoc trong vai phut")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    folds = (range(int(args.folds.split("-")[0]), int(args.folds.split("-")[1]) + 1)
             if "-" in args.folds else [int(x) for x in args.folds.split(",")])
    groups = (range(int(args.groups.split("-")[0]), int(args.groups.split("-")[1]) + 1)
              if "-" in args.groups else [int(x) for x in args.groups.split(",")])

    spec = COHORTS[args.cohort]
    markers = read_markers(str(spec["pos"]))
    pair_rows, ridge_rows = [], []

    for fold in folds:
        data = spec["data"] / f"fold{fold:02d}"
        for group in groups:
            cell = Path(args.sweep) / args.arm / f"fold{fold:02d}_g{group}" / "model"
            if not (cell / "manifest.json").exists():
                raise SystemExit(f"thieu model: {cell}")
            model = load_model(cell)
            net, cfg = model["net"], model["manifest"]["config"]
            labels = load_labels(str(spec["labels"]), GROUPS[group], n_digits=4)

            parts = {}
            for split in ("train", "val", "test"):
                ds = load_dataset(str(data / f"{split}.vcf.gz"), labels, markers, group,
                                  4, "train" if split == "train" else "test",
                                  cfg["phased"])
                x = torch.as_tensor(ds["data"], dtype=torch.float32)
                # [1] la lieu THAT 0/1/2 (`dosage`), khong phai nhan BCE da OR-collapse:
                # train_pair can dung so ban sao cho allele_weights/rare_mask/F1.
                parts[split] = (x, np.asarray(ds["dosage"], dtype=float), ds)

            blocks = _blocks(net.outputs_size)
            decoder = parts["test"][2]["encoder"].decoder
            samples = list(parts["test"][2]["sample-list"])

            # --- ridge tren z cua trunk CHUA tinh chinh -------------------
            # Thu tu quan trong: ban goc (dissect_rare_pathway.py) fit ridge tren
            # z cua mo hinh NEN, va blend_ridge_pair_cv tron ridge do voi du doan
            # pair. Neu fit sau khi pair da fine-tune trunk thi day la mot dai
            # luong khac, va hai tang khong con doc lap.
            with torch.no_grad():
                z = {s_: net.encode(parts[s_][0]).numpy()
                     for s_ in ("train", "val", "test")}
            tr, va, te = zscore(z["train"], z["val"], z["test"])
            ridge = np.zeros((len(samples), sum(s_ for _, _, s_ in blocks)))
            for gene, sl, size in blocks:
                ytr = (parts["train"][1][:, sl] > 0).astype(float)
                yva = (parts["val"][1][:, sl] > 0).astype(float)
                w, _ = select_lambda(tr, ytr, va, yva)
                ridge[:, sl] = _norm(apply_ridge(te, w)) * 2.0
            if args.only in ("both", "ridge"):
                _rows(ridge, samples, fold, blocks, decoder, ridge_rows)

            if args.only == "ridge":
                print(f"fold{fold:02d}_g{group} ridge xong", flush=True)
                continue

            # --- pair: fine-tune net+head tren train, dung som tren val --------
            head = PairEnergyHead(cfg["shared_dim"], [s_ for _, _, s_ in blocks])
            net, head, best = train_pair(
                net, head, parts["train"][0],
                torch.as_tensor(parts["train"][2]["label"], dtype=torch.float32),
                parts["train"][1], parts["val"][0], parts["val"][1],
                epochs=args.pair_epochs,
                log=lambda m, f=fold, g=group: print(f"fold{f:02d}_g{g} {m}", flush=True))
            _rows(predict_pair(net, head, parts["test"][0]), samples, fold, blocks,
                  decoder, pair_rows)

            print(f"fold{fold:02d}_g{group} xong (pair best rare F1 {best:.4f})",
                  flush=True)

    cols = ["sample_id", "fold", "gene", "allele", "dosage"]
    wanted = {"both": ("PAIR", "RIDGE_Z"), "pair": ("PAIR",), "ridge": ("RIDGE_Z",)}
    PRED.mkdir(parents=True, exist_ok=True)
    for rows, suffix in ((pair_rows, "PAIR"), (ridge_rows, "RIDGE_Z")):
        if suffix not in wanted[args.only]:
            continue
        out = PRED / f"{args.name}_{suffix}.dosage.csv.gz"
        pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
        print(f"{len(rows)} dong -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

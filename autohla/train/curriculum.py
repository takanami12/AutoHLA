"""Giai doan S2: curriculum 2-digit (rare_bce=0) -> 4-digit (rare_bce=cfg.rare_bce_max)
warm-start. Port cua hla_train + training +
trainer, gop hai lan goi hla_train.py (2-digit roi 4-digit,
noi bang --init-model) thanh MOT ham Python.

Nhanh `use_cross_validation` (mac dinh False trong hla_train.py, khong caller nao bat)
KHONG duoc port -- do la duong chet.

Hai giai doan trong ban goc la HAI PROCESS rieng, moi cai tu `set_seed(seed)` o dau
main; vi ham nay chay ca hai trong CUNG mot process, no phai TU GIEO LAI seed ngay
truoc moi giai doan (_seed_all) de mo phong dung "process moi", neu khong giai doan 2
se tieu thu RNG con lai cua giai doan 1 thay vi mot dong moi.

`model._train` = `model.train` cua AutoNet (AutoNet la nn.Module
thuan, khong co _train/_eval rieng nhu AENet -- FusionGNet._train chi lam
`self.train` roi lap lai tren tung HLA_Blocks, thua vi nn.Module.train da de quy
san). Goi DUY NHAT truoc vong lap epoch; test/eval trong _evaluate chuyen sang eval
mode va O LAI do cho ca epoch huan luyen ke tiep -- day KHONG phai loi, day la dieu
Khong di chuyen model.train vao trong vong lap: lam vay se bat lai dropout moi
epoch va lam thay doi ket qua allele hiem.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from autohla.io.dataset import load_dataset
from autohla.model.net import AutoNet

_BATCH_SIZE = 16
_LR = 1e-4
_PATIENCE = 7

# allele_threshold.json (ban goc) -- nguong homozygous_call CO DINH, dung de
# CHON checkpoint tot nhat (val_f1) trong luc train. Khac voi decode.tune_thresholds
# (do RIENG tren val cho tung lan chay, dung de xuat calls.csv cuoi cung o cong C4).
_HOMOZYGOUS_THRESHOLD = {
    "HLA_A": 0.06, "HLA_B": 0.055, "HLA_C": 0.025, "HLA_DPB1": 0.065,
    "HLA_DRB1": 0.055, "HLA_DQA1": 0.1, "HLA_DQB1": 0.075,
}


def _seed_all(seed):
    """Port data_helper (nhanh CPU, bo torch.cuda.manual_seed_all)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_2digit(labels):
    """Bang nhan 4-digit (2 truong "gia:protein") -> 2-digit (1 truong, chi gia dinh).

    Tuong duong load_labels(path, genes, n_digits=2) tren du lieu goc: khong allele
    nao trong cac file nhan cua du an nay vuot qua 2 truong, nen cat tiep tu ban da
    cat-4-digit cho ket qua HET giong cat thang tu file tho."""
    return labels.apply(lambda col: col.str.split(":").str[0])


def _load_compatible(model, source_state):
    """Port ban goc nnet (:146-153): nap moi tensor CUNG
    ten + CUNG shape. fc3 cua tung gene doi kich thuoc giua 2-digit/4-digit (vocab
    khac nhau) nen bi loai, giu nguyen gia tri khoi tao ngau nhien (da gieo seed
    lai) cua giai doan 4-digit -- dung y muon cua warm-start."""
    current = model.state_dict()
    compatible = {k: v for k, v in source_state.items()
                 if k in current and v.shape == current[k].shape}
    model.load_state_dict(compatible, strict=False)


def _make_batches(x, y, batch_size):
    """Port data_helper(mode='train'): cat lo THEO THU
    TU, gop lo cuoi vao lo truoc no neu no chi co 1 mau (BatchNorm1d o train mode
    can >=2 mau)."""
    batches = [(x[i:i + batch_size], y[i:i + batch_size])
              for i in range(0, len(x), batch_size)]
    if len(batches) > 1 and len(batches[-1][0]) == 1:
        last_x, last_y = batches.pop()
        prev_x, prev_y = batches.pop()
        batches.append((torch.cat([prev_x, last_x]), torch.cat([prev_y, last_y])))
    return batches


def _positive_weights(train_y, rare_bce_max):
    """Port AENet.set_train_label_frequency (:1624-1629). `train_y` la nhan
    multi-hot (logical-OR, khong qua 1 moi allele) nen `mean(0)` la TAN SUAT
    MANG (carrier fraction), khong phai tan suat ban sao."""
    if not rare_bce_max:
        return torch.ones(train_y.shape[1])
    freq = train_y.mean(0)
    return ((1 - freq) / freq.clamp_min(1 / len(train_y))).sqrt().clamp(1, rare_bce_max)


def _gene_scale(outputs_size, gene_weights):
    """Vector he so MOI COT theo GENE (None = toan 1, tuc khong doi hanh vi).

    Dung de HA TRONG SO mot gene trong loss ma VAN giu dau ra cua no: gene do
    van co head, van duoc du doan, chi bot keo trunk dung chung. Bo han gene ra
    khoi loss thi phai train mot mo hinh RIENG cho no -- day la cach tranh dieu do.
    """
    total = sum(size for _, size in outputs_size)
    scale = torch.ones(total)
    if not gene_weights:
        return scale
    start = 0
    for name, size in outputs_size:
        if name in gene_weights:
            scale[start:start + size] = float(gene_weights[name])
        start += size
    return scale


def _bce(output, target, positive_weights, rare_bce_max, gene_scale=None):
    """Port AENet.training_loss, duong song champion (khong recon/phase/haprec/
    hier/kd/mgda -- tat ca deu tat mac dinh.

    `gene_scale` (None = mac dinh) nhan them mot he so MOI COT theo gene. No nhan
    vao CA duong duong lan am, khac `positive_weights` chi cham o target > 0."""
    if rare_bce_max:
        weights = torch.where(target > 0, positive_weights, torch.ones_like(target))
    elif gene_scale is None:
        return F.binary_cross_entropy(output, target)
    else:
        weights = torch.ones_like(target)
    if gene_scale is not None:
        weights = weights * gene_scale
    return F.binary_cross_entropy(output, target, weight=weights)


def _evaluate(model, x, y, outputs_size):
    """Port trainer.test -- CHI val_loss/val_f1 (train_
    acc/precision/recall/etp khong anh huong checkpoint selection nen bo, xem
    tai lieu noi bo). Tung mau validation MOT, dung torch.argsort (khong
    phai numpy) de tie-break tren mang toan-0 khop bit-for-bit voi ban goc --
    day la thu quyet dinh checkpoint nao duoc luu."""
    val_loss = {name: 0.0 for name, _ in outputs_size}
    val_f1 = {name: 0.0 for name, _ in outputs_size}
    n = len(x)
    with torch.no_grad():
        for i in range(n):
            output = model(x[i:i + 1]).flatten(0)
            target = y[i]
            start = 0
            for name, size in outputs_size:
                out_block = output[start:start + size]
                tgt_block = target[start:start + size]
                val_loss[name] += float(F.binary_cross_entropy(out_block, tgt_block))

                allele_outs = out_block.argsort().numpy()[-2:][::-1].copy()
                allele_targets = tgt_block.argsort().numpy()[-2:][::-1]
                if float(out_block[int(allele_outs[1])]) < _HOMOZYGOUS_THRESHOLD[name]:
                    allele_outs[1] = allele_outs[0]
                y_pred = np.zeros(size)
                y_true = np.zeros(size)
                y_pred[allele_outs] = 1
                y_true[allele_targets] = 1
                s_true, s_pred = y_true.sum(), y_pred.sum()
                if s_true:
                    val_f1[name] += (2 * np.logical_and(y_true, y_pred).sum()
                                     / (s_true + s_pred))
                start += size
    for name, _ in outputs_size:
        val_loss[name] /= n
        val_f1[name] /= n
    return val_loss, val_f1


def _run_stage(model, train_x, train_y, val_x, val_y, outputs_size, *, epochs,
               rare_bce_max, gene_weights=None):
    """Port trainer.train (nhanh use_cross_validation=False)."""
    positive_weights = _positive_weights(train_y, rare_bce_max)
    gene_scale = None if not gene_weights else _gene_scale(outputs_size, gene_weights)
    optimizer = torch.optim.NAdam(model.parameters(), lr=_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=0)

    model.train()          # _train ngoai vong lap -- xem docstring dau file
    best_metric, best_val_loss, best_state, stale = 0.0, np.inf, None, 0
    eps = np.finfo(float).eps
    for _epoch in range(epochs):
        # shuffle_data mutate dataset['data'] TAI CHO -- moi epoch permutation moi
        # CHONG len ban da xao cua epoch truoc, khong phai ap vao thu tu goc.
        perm = np.random.permutation(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]
        for bx, by in _make_batches(train_x, train_y, _BATCH_SIZE):
            optimizer.zero_grad()
            loss = _bce(model(bx), by, positive_weights, rare_bce_max, gene_scale)
            loss.backward()
            optimizer.step()

        model.eval()        # o day den het ham -- epoch sau train tiep tren eval mode
        val_loss, val_f1 = _evaluate(model, val_x, val_y, outputs_size)
        val_loss_mean = float(np.mean(list(val_loss.values())))
        val_f1_mean = float(np.mean(list(val_f1.values())))
        scheduler.step(val_loss_mean)

        if val_f1_mean - eps > best_metric:
            best_metric, stale = val_f1_mean, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_f1_mean == best_metric and best_val_loss >= val_loss_mean:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_loss = val_loss_mean
        if best_val_loss < val_loss_mean:
            stale += 1
            if stale >= _PATIENCE:
                break
        else:
            stale = 0
            best_val_loss = val_loss_mean
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_curriculum(cfg, train_vcf, val_vcf, labels, markers, s1_path, out_dir, *,
                     epochs=100, device="cpu", train_samples=None, val_samples=None,
                     gene_weights=None):
    """2-digit (rare_bce=0) -> 4-digit (rare_bce=cfg.rare_bce_max) warm start.

    `labels`: bang nhan DAY DU cohort, chi gene cua cfg.group, DA cat ve 4-digit
    (vd `load_labels(FULL_LABEL_PATH, GROUPS[cfg.group], n_digits=4)`) -- vocab
    phai xay tu bang DAY DU, khong phai tap con cua fold. `s1_path`: checkpoint S1 THAM CHIEU cua ban goc (KHONG PHAI checkpoint
    AutoHLA tu huan luyen lai).
    """
    dev = torch.device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels_2 = _to_2digit(labels)
    # ---- giai doan 1: 2-digit, KHONG rare BCE, khoi dong tu S1 ----------------
    stage1_path = out / "2digit.pt"
    if stage1_path.exists():
        stage1_state = torch.load(stage1_path, map_location=dev)
    else:
        _seed_all(cfg.seed)
        train2 = load_dataset(train_vcf, labels_2, markers, cfg.group, 2, "train",
                              cfg.phased, keep_samples=train_samples)
        val2 = load_dataset(val_vcf, labels_2, markers, cfg.group, 2, "test",
                            cfg.phased, keep_samples=val_samples)
        model2 = AutoNet(train2["input-size"], train2["outputs-size"], cfg.group,
                         phased=cfg.phased, head=cfg.head,
                         shared_dim=cfg.shared_dim,
                         strides=cfg.strides, device=dev).to(dev)
        if s1_path is not None:
            model2.load_s1(s1_path)
        x2 = torch.as_tensor(train2["data"], dtype=torch.float32, device=dev)
        y2 = torch.as_tensor(train2["label"], dtype=torch.float32, device=dev)
        vx2 = torch.as_tensor(val2["data"], dtype=torch.float32, device=dev)
        vy2 = torch.as_tensor(val2["label"], dtype=torch.float32, device=dev)
        model2 = _run_stage(model2, x2, y2, vx2, vy2, train2["outputs-size"],
                            epochs=epochs, rare_bce_max=0.0,
                            gene_weights=gene_weights)
        stage1_state = model2.state_dict()
        torch.save(stage1_state, stage1_path)

    # ---- giai doan 2: 4-digit, warm-start tu giai doan 1 + rare BCE -----------
    # KHONG can model4.load_s1(s1_path): moi tensor no dat (backbone,
    # reconstruction_head) deu CUNG ten+shape voi giai doan 1 nen se bi
    # _load_compatible ghi de vo dieu kien ngay sau day -- goi no la lam viec chet.
    # load_dataset khong dung RNG nen nap truoc/sau _seed_all deu nhu nhau; nap o
    # day de biet input-size/outputs-size can cho ca hai nhanh duoi day.
    stage2_path = out / "4digit.pt"
    train4 = load_dataset(train_vcf, labels, markers, cfg.group, 4, "train",
                          cfg.phased, keep_samples=train_samples)
    val4 = load_dataset(val_vcf, labels, markers, cfg.group, 4, "test",
                        cfg.phased, keep_samples=val_samples)

    if stage2_path.exists():
        # Chi lai chet giua chung: da co ket qua giai doan 2 tren dia, khoi phai
        # huan luyen lai (dat nhat trong hai giai doan). RNG khong quan trong o
        # nhanh nay: gia tri khoi tao cua AutoNet bi load_state_dict ghi de het.
        model4 = AutoNet(train4["input-size"], train4["outputs-size"], cfg.group,
                         phased=cfg.phased, head=cfg.head,
                         shared_dim=cfg.shared_dim,
                         strides=cfg.strides, device=dev).to(dev)
        model4.load_state_dict(torch.load(stage2_path, map_location=dev))
    else:
        # _seed_all PHAI dung truoc AutoNet(...): thu tu tieu thu RNG cua construction
        # la thu C3 so bit-for-bit.
        _seed_all(cfg.seed)
        model4 = AutoNet(train4["input-size"], train4["outputs-size"], cfg.group,
                         phased=cfg.phased, head=cfg.head,
                         shared_dim=cfg.shared_dim,
                         strides=cfg.strides, device=dev).to(dev)
        _load_compatible(model4, stage1_state)
        x4 = torch.as_tensor(train4["data"], dtype=torch.float32, device=dev)
        y4 = torch.as_tensor(train4["label"], dtype=torch.float32, device=dev)
        vx4 = torch.as_tensor(val4["data"], dtype=torch.float32, device=dev)
        vy4 = torch.as_tensor(val4["label"], dtype=torch.float32, device=dev)
        model4 = _run_stage(model4, x4, y4, vx4, vy4, train4["outputs-size"],
                            epochs=epochs, rare_bce_max=cfg.rare_bce_max,
                            gene_weights=gene_weights)
        torch.save(model4.state_dict(), stage2_path)
    return model4

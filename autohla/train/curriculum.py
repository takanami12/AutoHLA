"""Giai doan S2: curriculum 2-digit (rare_bce=0) -> 4-digit (rare_bce=cfg.rare_bce_max)
warm-start. Port cua AEHLA/pipelines/hla_train.py + src/training.py +
objects/trainer.py::SingleTrainer, gop hai lan goi hla_train.py (2-digit roi 4-digit,
noi bang --init-model) thanh MOT ham Python.

Nhanh `use_cross_validation` (mac dinh False trong hla_train.py, khong caller nao bat)
KHONG duoc port -- do la duong chet, xem task-3-brief.md Step 5.

Hai giai doan trong AEHLA la HAI PROCESS rieng, moi cai tu `set_seed(seed)` o dau
main(); vi ham nay chay ca hai trong CUNG mot process, no phai TU GIEO LAI seed ngay
truoc moi giai doan (_seed_all) de mo phong dung "process moi", neu khong giai doan 2
se tieu thu RNG con lai cua giai doan 1 thay vi mot dong moi.

`model._train()` (CLAUDE.md) = `model.train()` cua AutoNet (AutoNet la nn.Module
thuan, khong co _train()/_eval() rieng nhu AENet -- FusionGNet._train() chi lam
`self.train()` roi lap lai tren tung HLA_Blocks, thua vi nn.Module.train() da de quy
san). Goi DUY NHAT truoc vong lap epoch; test()/eval trong _evaluate chuyen sang eval
mode va O LAI do cho ca epoch huan luyen ke tiep -- day KHONG phai loi, day la dieu
CLAUDE.md canh bao dung di chuyen: dua model.train() vao trong vong lap se bat lai
dropout moi epoch va lam sap F1 allele hiem (xem BAO_CAO_RARE_CURRICULUM.md).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from autohla.io.dataset import allele_frequencies, load_dataset
from autohla.model.net import AutoNet

_BATCH_SIZE = 16
_LR = 1e-4
_PATIENCE = 7

# configs/allele_threshold.json (AEHLA) -- nguong homozygous_call CO DINH, dung de
# CHON checkpoint tot nhat (val_f1) trong luc train. Khac voi decode.tune_thresholds
# (do RIENG tren val cho tung lan chay, dung de xuat calls.csv cuoi cung o cong C4).
_HOMOZYGOUS_THRESHOLD = {
    "HLA_A": 0.06, "HLA_B": 0.055, "HLA_C": 0.025, "HLA_DPB1": 0.065,
    "HLA_DRB1": 0.055, "HLA_DQA1": 0.1, "HLA_DQB1": 0.075,
}


def _seed_all(seed):
    """Port src/data_helper.py::set_seed (nhanh CPU, bo torch.cuda.manual_seed_all)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_2digit(labels):
    """Bang nhan 4-digit (2 truong "gia:protein") -> 2-digit (1 truong, chi gia dinh).

    Tuong duong load_labels(path, genes, n_digits=2) tren du lieu goc: khong allele
    nao trong cac file nhan cua du an nay vuot qua 2 truong, nen cat tiep tu ban da
    cat-4-digit cho ket qua HET giong cat thang tu file tho (xem task-3-report.md
    muc 8)."""
    return labels.apply(lambda col: col.str.split(":").str[0])


def _load_compatible(model, source_state):
    """Port AEHLA models/nnet.py::load_compatible (:146-153): nap moi tensor CUNG
    ten + CUNG shape. fc3 cua tung gene doi kich thuoc giua 2-digit/4-digit (vocab
    khac nhau) nen bi loai, giu nguyen gia tri khoi tao ngau nhien (da gieo seed
    lai) cua giai doan 4-digit -- dung y muon cua warm-start."""
    current = model.state_dict()
    compatible = {k: v for k, v in source_state.items()
                 if k in current and v.shape == current[k].shape}
    model.load_state_dict(compatible, strict=False)


def _make_batches(x, y, batch_size, mask=None):
    """Port src/data_helper.py::transform_dataset(mode='train'): cat lo THEO THU
    TU, gop lo cuoi vao lo truoc no neu no chi co 1 mau (BatchNorm1d o train mode
    can >=2 mau)."""
    arrays = (x, y) if mask is None else (x, y, mask)
    batches = [tuple(a[i:i + batch_size] for a in arrays)
              for i in range(0, len(x), batch_size)]
    if len(batches) > 1 and len(batches[-1][0]) == 1:
        last, previous = batches.pop(), batches.pop()
        batches.append(tuple(torch.cat([a, b]) for a, b in zip(previous, last)))
    return batches


def _positive_weights(train_dosage, rare_bce_max, sizes):
    """Rare BCE weights from valid train copies of each gene, not carriers."""
    if not rare_bce_max:
        return train_dosage.new_ones(train_dosage.shape[1])
    af = train_dosage.new_tensor(allele_frequencies(train_dosage.detach().cpu().numpy(), sizes))
    return ((1 - af) / af.clamp_min(torch.finfo(af.dtype).eps)).sqrt().clamp(1, rare_bce_max)


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


def _bce(output, target, positive_weights, rare_bce_max, gene_scale=None, mask=None):
    """Port AENet.training_loss, duong song champion (khong recon/phase/haprec/
    hier/kd/mgda -- tat ca deu tat mac dinh, xem task-3-report.md muc 3).

    `gene_scale` (None = mac dinh) nhan them mot he so MOI COT theo gene. No nhan
    vao CA duong duong lan am, khac `positive_weights` chi cham o target > 0."""
    if rare_bce_max:
        weights = torch.where(target > 0, positive_weights, torch.ones_like(target))
    elif gene_scale is None and mask is None:
        return F.binary_cross_entropy(output, target)
    else:
        weights = torch.ones_like(target)
    if gene_scale is not None:
        weights = weights * gene_scale
    if mask is None:
        return F.binary_cross_entropy(output, target, weight=weights)
    loss = F.binary_cross_entropy(output, target, weight=weights, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def _evaluate(model, x, y, outputs_size, *, dosage, label_mask=None,
              valid_copies=None):
    """Checkpoint F1 from pooled true diploid copies, including unseen FN.

    Missing genotypes do not enter evaluation. BCE additionally excludes
    genotypes outside the training vocabulary, whose full target is unknown.
    """
    val_loss = {name: 0.0 for name, _ in outputs_size}
    loss_count = dict.fromkeys(val_loss, 0)
    counts = {name: [0.0, 0.0] for name in val_loss}
    with torch.no_grad():
        for i in range(len(x)):
            output = model(x[i:i + 1]).flatten(0)
            start = 0
            for gene, (name, size) in enumerate(outputs_size):
                out_block = output[start:start + size]
                tgt_block = y[i, start:start + size]
                true_block = dosage[i, start:start + size]
                known = (bool(label_mask[i, start:start + size].all())
                         if label_mask is not None else float(true_block.sum()) == 2)
                if known:
                    val_loss[name] += float(F.binary_cross_entropy(out_block, tgt_block))
                    loss_count[name] += 1
                valid = (float(valid_copies[i, gene]) if valid_copies is not None
                         else float(true_block.sum()))
                if valid == 2:
                    order = out_block.argsort(descending=True)
                    first = int(order[0])
                    second = int(order[1]) if size > 1 else first
                    if float(out_block[second]) < _HOMOZYGOUS_THRESHOLD[name]:
                        second = first
                    pred = torch.zeros_like(true_block)
                    pred[first] += 1
                    pred[second] += 1
                    counts[name][0] += float(torch.minimum(pred, true_block).sum())
                    counts[name][1] += valid + float(pred.sum())
                start += size
    if not sum(den for _, den in counts.values()):
        raise ValueError("validation has no complete HLA genotypes")
    val_f1 = {}
    for name in val_loss:
        val_loss[name] /= max(loss_count[name], 1)
        tp, den = counts[name]
        val_f1[name] = 2 * tp / max(den, 1)
    val_f1["micro"] = (2 * sum(tp for tp, _ in counts.values())
                       / sum(den for _, den in counts.values()))
    return val_loss, val_f1


def _run_stage(model, train_x, train_y, val_x, val_y, outputs_size, *, epochs,
               rare_bce_max, train_dosage, gene_weights=None, train_mask=None, val_dosage=None,
               val_mask=None, val_valid_copies=None):
    """Port objects/trainer.py::SingleTrainer.train() (nhanh use_cross_validation=False)."""
    if val_dosage is None:
        raise ValueError("checkpoint evaluation requires true 0/1/2 dosage")
    train_mask = (torch.ones_like(train_y, dtype=torch.bool) if train_mask is None
                  else torch.as_tensor(train_mask, dtype=torch.bool, device=train_y.device))
    val_dosage = torch.as_tensor(val_dosage, dtype=val_y.dtype, device=val_y.device)
    val_mask = (None if val_mask is None else
                torch.as_tensor(val_mask, dtype=torch.bool, device=val_y.device))
    train_dosage = torch.as_tensor(train_dosage, dtype=train_y.dtype, device=train_y.device)
    positive_weights = _positive_weights(train_dosage, rare_bce_max,
                                           [size for _, size in outputs_size])
    gene_scale = (None if not gene_weights else
                  _gene_scale(outputs_size, gene_weights).to(train_y.device))
    optimizer = torch.optim.NAdam(model.parameters(), lr=_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=0)

    model.train()          # _train() ngoai vong lap -- xem docstring dau file
    best_metric, best_val_loss, best_state, stale = 0.0, np.inf, None, 0
    eps = np.finfo(float).eps
    for _epoch in range(epochs):
        # shuffle_data mutate dataset['data'] TAI CHO -- moi epoch permutation moi
        # CHONG len ban da xao cua epoch truoc, khong phai ap vao thu tu goc.
        perm = np.random.permutation(len(train_x))
        train_x, train_y, train_mask = train_x[perm], train_y[perm], train_mask[perm]
        for bx, by, bm in _make_batches(train_x, train_y, _BATCH_SIZE, train_mask):
            optimizer.zero_grad()
            loss = _bce(model(bx), by, positive_weights, rare_bce_max, gene_scale, bm)
            loss.backward()
            optimizer.step()

        model.eval()        # o day den het ham -- epoch sau train tiep tren eval mode
        val_loss, val_f1 = _evaluate(model, val_x, val_y, outputs_size,
                                     dosage=val_dosage, label_mask=val_mask,
                                     valid_copies=val_valid_copies)
        val_loss_mean = float(np.mean(list(val_loss.values())))
        val_f1_mean = val_f1["micro"]
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

    `labels`: normalized 4-digit labels. Vocabulary uses training samples
    only; validation reuses that encoder and retains unseen copies as FN.
    The returned network exposes the fitted `encoder`.
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
                            cfg.phased, keep_samples=val_samples, encoder=train2["encoder"])
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
                            gene_weights=gene_weights, train_dosage=train2["dosage"],
                            train_mask=train2["label-mask"],
                            val_dosage=val2["dosage"], val_mask=val2["label-mask"],
                            val_valid_copies=val2["valid-copies"])
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
                        cfg.phased, keep_samples=val_samples, encoder=train4["encoder"])

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
        # la thu C3 so bit-for-bit (xem task-3-report.md).
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
                            gene_weights=gene_weights, train_dosage=train4["dosage"],
                            train_mask=train4["label-mask"],
                            val_dosage=val4["dosage"], val_mask=val4["label-mask"],
                            val_valid_copies=val4["valid-copies"])
        torch.save(model4.state_dict(), stage2_path)
    model4.encoder = train4["encoder"]
    return model4

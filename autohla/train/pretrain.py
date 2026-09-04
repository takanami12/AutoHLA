"""S1: port cua s1_pretrain, chi nhanh nap du lieu KHONG NHAN
(mode="unlabeled") -- xem docstring cua pretrain_s1 cho ly do van con
`_rng_compat_outputs_size` du day la duong du lieu khong nhan.

Fact 1 (task-2 brief): S1 LUON huan luyen KHONG pha, ke ca cho arm phased --
`run_lphase_group_cv.sh` tat AE_LPHASE/AE_LPHASE_ORACLE truoc khi
goi s1_pretrain.py, vi bat len se doi dinh dang dataset va lam arm base im lang
huan luyen tren dinh dang khac. Vi vay `pretrain_s1` khong nhan tham so phased nao
va luon goi `load_dataset(..., mode="unlabeled", phased=False)`.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F

from autohla.io.dataset import load_dataset
from autohla.model.net import AutoNet

_BATCH_SIZE = 128
_LR = 1e-3
_WEIGHT_DECAY = 1e-2
_WARMUP_STEPS = 100
_GRAD_CLIP = 1.0
_PATIENCE = 8


def _epoch_pass(model, x, optimizer=None, scheduler=None):
    """Mot luot qua x; huan luyen khi co optimizer, nguoc lai chi danh gia."""
    losses, accs = [], []
    order = torch.randperm(len(x)) if optimizer is not None else torch.arange(len(x))
    for i in range(0, len(x), _BATCH_SIZE):
        batch = x[order[i:i + _BATCH_SIZE]]
        dense, _ = model.dense_input(batch)
        corrupted, masked = model.corrupt(dense)
        if not masked.any():
            continue
        with torch.set_grad_enabled(optimizer is not None):
            logits = model.recon_forward(corrupted)
            target = dense[:, 0].long()[masked]
            prediction = logits.argmax(1)[masked]
            loss = F.cross_entropy(logits.permute(0, 2, 1)[masked], target)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            optimizer.step()
            scheduler.step()
        losses.append(loss.item())
        accs.append(prediction.eq(target).float().mean().item())
    return float(np.mean(losses)), float(np.mean(accs))


def pretrain_s1(train_vcf, val_vcf, markers, group, out_path, *, epochs=100,
                seed=77, threads=2, device="cpu", head="full",
                _rng_compat_outputs_size=()):
    """Huan luyen backbone+reconstruction_head bang pretext che-doan-lai-dosage.

    Luu y thu tu goi (khop s1_pretrain de RNG tieu thu
    dung thu tu): seed -> so luong -> nap du lieu (khong RNG) -> dung AutoNet (RNG:
    backbone, reconstruction_head, shared, roi HLA_Blocks NEU
    _rng_compat_outputs_size khong rong) -> vong lap epoch (RNG: randperm moi
    epoch train, rand_like moi batch ca train lan eval, vi corrupt() chay khong
    dieu kien toi optimizer).

    `_rng_compat_outputs_size` KHONG PHAI mot tham so mo hinh -- no khong duoc
    dat ten `outputs_size` vi ly do do: doi gia tri cua no KHONG doi S1 hoc gi
    ("hoc gi" nghia la khong ai, ke ca AutoNet.forward(), tung goi toi HLA_Blocks
    hay self.shared -- S1 chi goi dense_input/corrupt/recon_forward). Tac dung
    DUY NHAT cua no la mot gia tri du lieu tuong thich RNG: AutoNet.__init__ rut
    so tu RNG cho HLA_Blocks NGAY SAU backbone+reconstruction_head, va vi RNG la
    MOT DONG CHUNG, so luong rut o do doi VI TRI xuat phat cua randperm/rand_like
    trong vong lap epoch ngay ben duoi -- tuc doi ca duong huan luyen di theo
    huong khac, DU backbone+reconstruction_head tai thoi diem khoi tao khong doi.
    Day la mot compat shim de tai lap dung bit-for-bit checkpoint S1 cua ban goc
    (xem ghi chu duoi), KHONG PHAI mot nut chinh kien truc: dung dat ten cong
    khai, dung dua no vao CLI/RunConfig cua cac task sau.

    Ly do no ton tai: VN1K's S1 that KHONG dung --unlabeled
    (run_lphase_group_cv.sh bo co do) -- no di qua duong nap co
    nhan (nhung bo nhan luc train), nen outputs-size THAT cua no la kich thuoc
    tu vung allele cua tung gene, khong rong. Ben goi (run_s1_sweep)
    truyen dung gia tri do cho VN1K va rong cho HAN (HAN that su dung
    --unlabeled, nen outputs-size that cua no da la rong san).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads:
        torch.set_num_threads(threads)

    dev = torch.device(device)
    trainset = load_dataset(train_vcf, labels=None, markers=markers, group=group,
                            n_digits=4, mode="unlabeled", phased=False)
    valset = load_dataset(val_vcf, labels=None, markers=markers, group=group,
                          n_digits=4, mode="unlabeled", phased=False) \
        if val_vcf else trainset
    x_train = torch.as_tensor(trainset["data"], dtype=torch.float32, device=dev)
    x_val = torch.as_tensor(valset["data"], dtype=torch.float32, device=dev)

    model = AutoNet(trainset["input-size"], list(_rng_compat_outputs_size), group,
                    device=dev, head=head).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / _WARMUP_STEPS))

    best, stale = -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_acc = _epoch_pass(model, x_train, optimizer, scheduler)
        model.eval()
        val_loss, val_acc = _epoch_pass(model, x_val)
        print("S1 epoch={:03d} train_loss={:.4f} train_acc={:.4f} "
              "val_loss={:.4f} val_acc={:.4f}".format(
                  epoch, train_loss, train_acc, val_loss, val_acc), flush=True)
        if val_acc > best + 1e-4:
            best, stale = val_acc, 0
            model.save_s1(out_path)
        else:
            stale += 1
            if stale >= _PATIENCE:
                print("S1 early stopping at epoch {}; best_val_acc={:.4f}"
                      .format(epoch, best), flush=True)
                break
    # flush=True nhu cac dong epoch: khong co no, hai dong ket thuc S1 nam lai
    # trong buffer va log trong nhu bi treo trong suot giai doan curriculum (von
    # khong in gi theo epoch). Da mat mot lan chan doan nham vi cho nay.
    print("S1 saved to {}; best val recon acc {:.4f}".format(out_path, best),
          flush=True)

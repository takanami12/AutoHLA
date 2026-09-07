"""Huan luyen tang pair: fine-tune CA mo hinh voi loss phu tren cap khong thu tu.

Port cua run_pair_energy_cv. Ba dieu de hieu sai neu chi doc
model/pair.py:

1. Day KHONG phai mot head gan tren trunk dong bang. Optimizer nhan CA tham so
   goc (lr 1e-4) lan tham so head (lr 5e-4), va loss la
   `BCE(base) + lambda * pair_loss` voi lambda = 0.2. Trunk dich theo.
2. Trong so allele la sqrt((1-af)/af) cat trong [1, 10], tinh tu train cua chinh
   fold do -- cung ho voi rare-weighted BCE cua curriculum.
3. Dung som theo F1 cua rieng bin `<1%` tren val, KHONG phai theo loss. Mot lan
   chay tot o bin pho bien nhung te o bin hiem se bi bo.

Ba nhanh da bi bac bo khong duoc port (xem model/pair.py): MLP scorer, pair
prior, SWA.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

_EPOCHS = 40
_BATCH = 32
_LAMBDA = 0.2
_PATIENCE = 7
_BASE_LR = 1e-4
_HEAD_LR = 5e-4


def pair_targets(dosage, sizes, head):
    """Lieu that 0/1/2 -> chi so lop trong bang cap khong thu tu.

    Dong hop (mot allele hai ban sao) phai ra cap duong cheo (i, i): sau khi lay
    allele cao nhat ra, phan con lai tong bang 0 nen `second` quay ve `first`.
    """
    targets, start = [], 0
    rows = torch.arange(dosage.shape[0], device=dosage.device)
    for gene, size in enumerate(sizes):
        block = dosage[:, start:start + size]
        first = block.argmax(dim=1)
        rest = block.clone()
        rest[rows, first] = 0
        second = rest.argmax(dim=1)
        second = torch.where(rest.sum(dim=1) > 0, second, first)
        lookup = getattr(head, f"pair_lookup_{gene}").to(dosage.device)
        target = lookup[first, second]
        if (target < 0).any():
            raise ValueError("pair target is not in the unordered-pair table")
        targets.append(target)
        start += size
    return targets


def weighted_pair_loss(logits, targets, head, allele_weights):
    """Cross-entropy tren cap, trong so theo do hiem cua HAI allele trong cap,
    chuan hoa theo log so lop de cac gene co so allele khac nhau van so duoc."""
    losses, start = [], 0
    for gene, (logit, target, size) in enumerate(zip(logits, targets, head.sizes)):
        i, j = head.pair_indices(gene, logit.device)
        pair_weight = (allele_weights[start + i] + allele_weights[start + j]) / 2
        sample_weight = pair_weight[target]
        nll = F.cross_entropy(logit, target, reduction="none")
        loss = (nll * sample_weight).sum() / sample_weight.sum().clamp_min(1)
        losses.append(loss / math.log(max(logit.shape[1], 2)))
        start += size
    return torch.stack(losses).mean()


def allele_weights_from(train_truth):
    """sqrt((1-af)/af) cat trong [1, 10] -- cung ho voi rare-weighted BCE."""
    af = train_truth.sum(0) / (2 * len(train_truth))
    return np.sqrt((1 - af) / np.maximum(af, 1 / len(train_truth))).clip(1, 10)


def rare_mask(train_truth, threshold=0.01):
    af = train_truth.sum(0) / (2 * len(train_truth))
    return (af > 0) & (af < threshold)


def _micro_f1(pred, truth, mask):
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    tp = np.minimum(pred, truth).sum()
    t, p = truth.sum(), pred.sum()
    sn, ppv = tp / max(t, 1), tp / max(p, 1)
    return 0.0 if sn + ppv == 0 else 2 * sn * ppv / (sn + ppv)


def predict_pair(net, head, x, decode=None):
    net.eval()
    with torch.no_grad():
        score = net(x)
        _, _, dosage = head(net.encode(x), score)
    dosage = torch.cat(dosage, dim=1).numpy()
    return dosage if decode is None else decode(dosage)


def train_pair(net, head, train_x, train_y, train_truth, val_x, val_truth, *,
               epochs=_EPOCHS, batch_size=_BATCH, pair_lambda=_LAMBDA,
               patience=_PATIENCE, seed=77, log=print):
    """Fine-tune (net, head) chung. Tra (net, head) o trang thai TOT NHAT theo F1
    bin `<1%` tren val.

    `train_y` la nhan multi-hot cho BCE; `train_truth`/`val_truth` la lieu THAT
    0/1/2 (dung cho nhan cap, trong so allele, va thuoc do dung som).
    """
    torch.manual_seed(seed)
    targets = pair_targets(torch.as_tensor(train_truth, dtype=torch.float32),
                           head.sizes, head)
    weights = torch.as_tensor(allele_weights_from(train_truth), dtype=torch.float32)
    mask = rare_mask(train_truth)
    optimizer = torch.optim.NAdam(
        [{"params": list(net.parameters()), "lr": _BASE_LR},
         {"params": list(head.parameters()), "lr": _HEAD_LR}], lr=_BASE_LR)

    best, best_state, stale = -np.inf, None, 0
    # net.train NGOAI vong epoch -- dua vao trong bat dropout luc danh gia va
    # lam F1 bin `<1%` sap ve 0 (do noi bo).
    net.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_x))
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        # Lo cuoi chi 1 mau -> gop vao lo truoc no: BatchNorm1d o train mode doi
        # >=2 hang ("Expected more than 1 value per channel"). Cung luat voi
        # curriculum._make_batches, va no CHI cham vao truong hop do -- moi split
        # co du 2 mau o lo cuoi giu nguyen tung buoc gradient.
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat([batches[-1], last])
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            score = net(train_x[batch])
            logits, _, _ = head(net.encode(train_x[batch]), score)
            base = F.binary_cross_entropy(score.clamp(1e-7, 1 - 1e-7), train_y[batch])
            pair = weighted_pair_loss(logits, [t[batch] for t in targets], head, weights)
            (base + pair_lambda * pair).backward()
            optimizer.step()
        val_f1 = _micro_f1(predict_pair(net, head, val_x), val_truth, mask)
        net.train()
        log(f"pair epoch={epoch:02d} val_rare_f1={val_f1:.5f}")
        if val_f1 > best + 1e-12:
            best, stale = val_f1, 0
            best_state = (copy.deepcopy(net.state_dict()),
                          copy.deepcopy(head.state_dict()))
        else:
            stale += 1
            if stale >= patience:
                log(f"pair early stopping at epoch {epoch}; best_rare_f1={best:.5f}")
                break
    if best_state is not None:
        net.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    net.eval()
    return net, head, float(best)

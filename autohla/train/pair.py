"""Huan luyen tang pair: fine-tune CA mo hinh voi loss phu tren cap khong thu tu.

Port cua scripts/eval/run_pair_energy_cv.py. Ba dieu de hieu sai neu chi doc
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

from autohla.decode import map_diploid, to_prob
from autohla.io.dataset import allele_frequencies
from autohla.train.curriculum import _bce

_EPOCHS = 40
_BATCH = 32
_LAMBDA = 0.2
_PATIENCE = 7
_BASE_LR = 1e-4
_HEAD_LR = 5e-4


def pair_targets(dosage, sizes, head, mask=None):
    """Lieu that 0/1/2 -> chi so lop trong bang cap khong thu tu.

    Homozygotes target (i, i); incomplete or masked genotypes use ignore -100.
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
        valid = block.sum(dim=1) == 2
        if mask is not None:
            valid &= mask[:, start:start + size].all(dim=1)
        targets.append(torch.where(valid, target, -100))
        start += size
    return targets


def weighted_pair_loss(logits, targets, head, allele_weights):
    """Cross-entropy tren cap, trong so theo do hiem cua HAI allele trong cap,
    chuan hoa theo log so lop de cac gene co so allele khac nhau van so duoc."""
    losses, start = [], 0
    for gene, (logit, target, size) in enumerate(zip(logits, targets, head.sizes)):
        valid = target >= 0
        i, j = head.pair_indices(gene, logit.device)
        pair_weight = (allele_weights[start + i] + allele_weights[start + j]) / 2
        start += size
        if not valid.any():
            continue
        sample_weight = pair_weight[target[valid]]
        nll = F.cross_entropy(logit[valid], target[valid], reduction="none")
        loss = (nll * sample_weight).sum() / sample_weight.sum().clamp_min(1)
        losses.append(loss / math.log(max(logit.shape[1], 2)))
    return (torch.stack(losses).mean() if losses else
            sum(logit.sum() * 0 for logit in logits))


def allele_weights_from(train_truth, sizes=None):
    """sqrt((1-af)/af) cat trong [1, 10] -- cung ho voi rare-weighted BCE."""
    af = allele_frequencies(train_truth, sizes or [train_truth.shape[1]])
    return np.sqrt((1 - af) / np.maximum(af, 1 / max(len(train_truth), 1))).clip(1, 10)


def rare_mask(train_truth, threshold=0.01, sizes=None):
    af = allele_frequencies(train_truth, sizes or [train_truth.shape[1]])
    return (af > 0) & (af < threshold)


def _micro_f1(pred, truth, mask, unseen_copies=0):
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    tp = np.minimum(pred, truth).sum()
    t, p = truth.sum() + unseen_copies, pred.sum()
    sn, ppv = tp / max(t, 1), tp / max(p, 1)
    return 0.0 if sn + ppv == 0 else 2 * sn * ppv / (sn + ppv)


def predict_pair(net, head, x, decode=None):
    net.eval()
    head.eval()
    with torch.no_grad():
        score, shared = net.forward_with_embedding(x)
        _, _, dosage = head(shared, score)
    dosage = torch.cat(dosage, dim=1).cpu().numpy()
    return dosage if decode is None else decode(dosage)


def train_pair(net, head, train_x, train_y, train_truth, val_x, val_truth, *,
               epochs=_EPOCHS, batch_size=_BATCH, pair_lambda=_LAMBDA,
               patience=_PATIENCE, seed=77, log=print, train_mask=None,
               val_valid_copies=None):
    """Fine-tune (net, head) chung. Tra (net, head) o trang thai TOT NHAT theo F1
    bin `<1%` tren val.

    `train_y` la nhan multi-hot cho BCE; `train_truth`/`val_truth` la lieu THAT
    0/1/2 (dung cho nhan cap, trong so allele, va thuoc do dung som).
    """
    torch.manual_seed(seed)
    if epochs < 1 or len(train_x) < 2:
        raise ValueError("pair training needs positive epochs and at least two samples")
    if torch.any((train_y < 0) | (train_y > 1)):
        raise ValueError("train_y must be binary carrier targets, not copy dosage")
    device = train_x.device
    truth_tensor = torch.as_tensor(train_truth, dtype=torch.float32, device=device)
    if train_mask is None:
        train_mask = np.concatenate([
            np.repeat((block.sum(1) == 2)[:, None], block.shape[1], axis=1)
            for block in np.split(train_truth, np.cumsum(head.sizes)[:-1], axis=1)], axis=1)
    train_mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    targets = pair_targets(truth_tensor, head.sizes, head, train_mask)
    weights = torch.as_tensor(allele_weights_from(train_truth, head.sizes),
                              dtype=torch.float32, device=device)
    mask = rare_mask(train_truth, sizes=head.sizes)
    val_truth = np.asarray(val_truth)
    valid = np.zeros_like(val_truth, dtype=bool)
    unseen, start = 0, 0
    for gene, size in enumerate(head.sizes):
        known = val_truth[:, start:start + size].sum(1)
        copies = known if val_valid_copies is None else np.asarray(val_valid_copies)[:, gene]
        valid[:, start:start + size] = (copies == 2)[:, None]
        unseen += (copies - known)[copies == 2].sum()
        start += size
    # A tiny reference can contain no rare validation copies; select on all
    # valid copies in that case instead of accepting the first epoch blindly.
    if (val_truth * valid)[:, mask].sum() + unseen == 0:
        mask = None
    optimizer = torch.optim.NAdam(
        [{"params": list(net.parameters()), "lr": _BASE_LR},
         {"params": list(head.parameters()), "lr": _HEAD_LR}], lr=_BASE_LR)

    best, best_state, stale = -np.inf, None, 0
    # net.train() NGOAI vong epoch -- dua vao trong bat dropout luc danh gia va
    # lam F1 bin `<1%` sap ve 0 (memory `dropout-eval-mode-rare-collapse`).
    net.train()
    head.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_x), device=device)
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
            score, shared = net.forward_with_embedding(train_x[batch])
            logits, _, _ = head(shared, score)
            base = _bce(score.clamp(1e-7, 1 - 1e-7), train_y[batch],
                        weights, 0, mask=train_mask[batch])
            pair = weighted_pair_loss(logits, [t[batch] for t in targets], head, weights)
            (base + pair_lambda * pair).backward()
            optimizer.step()
        predicted = predict_pair(net, head, val_x)
        calls = np.concatenate([map_diploid(to_prob(block)) for block in
                                np.split(predicted, np.cumsum(head.sizes)[:-1], axis=1)], axis=1)
        val_f1 = _micro_f1(calls * valid, val_truth * valid, mask, unseen)
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
    head.eval()
    return net, head, float(best)

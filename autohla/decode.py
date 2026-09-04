"""Bo giai ma lieu mem -> lieu cung. BA bo giai ma khac nhau ton tai trong repo nay
(da tung gay nham lan): `export_calls` dung top-2 + nguong moi gene do
tren val; `blend_ridge_pair_cv` dung MAP tren cap khong thu tu
(`map_diploid`); `make_hardcall_pools` dung top-2 tran (khong nguong).

`map_diploid` la bo giai ma MAC DINH cua goi nay. `threshold_calls`/`tune_thresholds`
chi ton tai de tai lap dung so cu cua `export_calls` (cong C4).
"""
import itertools

import numpy as np
import torch


def map_diploid_pairs(prob: np.ndarray, topk: int = 5):
    """MAP tren cap KHONG THU TU (i, j), i<=j, tren top-k allele theo prob.

    P(dong hop i) = p_i^2, P(di hop i,j) = 2 p_i p_j (giong homozygous_call trong
    trainer). Tra (chi_so_cap, posterior): chi_so_cap la (N, 2)
    int, posterior la (N,) float trong [0, 1] -- xac suat cua CAP DA CHON, dung
    duoc vi prob moi hang da chuan hoa tong 1 (to_prob) nen tong xac suat tren toan
    bo khong gian cap dung bang 1.

    Chep nguyen van tu blend_ridge_pair_cv (`map_diploid`),
    tach rieng phan toi da hoa; `map_diploid` ben duoi goi lai ham nay nen hai ham
    khong bao gio bat dong (xem test_pairs_and_dosage_agree).
    """
    n = len(prob)
    pairs = np.zeros((n, 2), dtype=int)
    posterior = np.zeros(n, dtype=float)
    for i in range(n):
        top = np.argsort(-prob[i])[:topk]
        logp = np.log(np.maximum(prob[i][top], 1e-12))
        best, arg = -np.inf, (top[0], top[0])
        for u, v in itertools.combinations_with_replacement(range(len(top)), 2):
            value = logp[u] + logp[v] + (np.log(2) if u != v else 0.0)
            if value > best:
                best, arg = value, (top[u], top[v])
        pairs[i] = arg
        posterior[i] = np.exp(best)
    return pairs, posterior


def map_diploid(prob: np.ndarray, topk: int = 5) -> np.ndarray:
    """MAP tren cap khong thu tu -> ma tran lieu 0/1/2. Bo giai ma MAC DINH.

    Chu ky GIU Y NGUYEN ban goc (blend_ridge_pair_cv)
    de cong C4 so duoc: cung mot dau vao ra cung mot dau ra, xem
    test_matches_the_blend_script_on_random_input.
    """
    pairs, _ = map_diploid_pairs(prob, topk)
    out = np.zeros_like(prob)
    rows = np.arange(len(prob))
    out[rows, pairs[:, 0]] += 1
    out[rows, pairs[:, 1]] += 1
    return out


def to_prob(matrix: np.ndarray) -> np.ndarray:
    """Dosage/score khong am -> phan bo xac suat moi hang tong 1.

    Chep nguyen van tu blend_ridge_pair_cv.
    """
    m = np.clip(matrix, 0.0, None) + 1e-6
    return m / m.sum(1, keepdims=True)


def _top2(block) -> np.ndarray:
    """Hai chi so diem cao nhat moi hang -- CHEP NGUYEN VAN bieu thuc torch cua
    ban goc `block.argsort(dim=1)[:, -2:].flip(1)` (export_calls).

    Khong viet lai bang numpy. `torch.argsort` mac dinh KHONG on dinh, va khi hai
    allele bang nhau TUNG BIT (sigmoid bao hoa, xay ra that o HAN) thu tu no tra
    ve khong theo quy tac chi-so-nho-truoc hay chi-so-lon-truoc nao ca -- no tat
    dinh theo du lieu chu khong theo mot luat viet lai duoc. Do tren han g4
    (3.147 o mau x gene): bieu thuc torch sai 0, np.argsort(-block, kind="stable")
    sai 1, np.argsort(block, kind="stable")[:, -2:][:, ::-1] sai 2.
    """
    return torch.as_tensor(block).argsort(dim=1)[:, -2:].flip(1).numpy()


def threshold_calls(scores: np.ndarray, thresholds: dict, outputs_size) -> np.ndarray:
    """Top-2 + nguong moi gene (dong hop khi allele thu 2 duoi nguong cua gene).

    CHI de tai lap so cu cua export_calls (cong C4) -- KHONG phai bo giai ma
    mac dinh cua goi nay (dung `map_diploid`). `scores` la ma tran sigmoid da noi
    het cac gene (N, tong_allele); `thresholds` la {ten_gene: nguong} tu
    `tune_thresholds`; `outputs_size` la [(ten_gene, kich_thuoc), ...] cung thu tu
    voi cac khoi trong `scores` (vd `AutoNet.outputs_size`).

    Port cua export_calls (nhanh khong-haprec), chuyen tu
    torch sang numpy.
    """
    out = np.zeros_like(scores)
    start = 0
    for name, size in outputs_size:
        block = scores[:, start:start + size]
        top2 = _top2(block)
        rows = np.arange(len(block))
        first, second = top2[:, 0], top2[:, 1]
        het = block[rows, second] >= thresholds[name]
        out[rows, start + first] += 1
        out[rows, start + np.where(het, second, first)] += 1
        start += size
    return out


def tune_thresholds(scores: np.ndarray, truth: np.ndarray, outputs_size) -> dict:
    """Nguong moi gene toi da hoa F1 tren val (top-2 + nguong), cung thu tuc
    export_calls dung de sinh calls.csv. `truth` la dosage THAT 0/1/2 (khong
    phai nhan multi-hot BCE) -- xem `label_dosage`-style constructor o noi goi.

    Port cua export_calls, chuyen tu torch
    sang numpy (khong can model/dataset song, chi can scores+truth da tinh san).
    """
    thresholds, start = {}, 0
    grid = np.arange(100) / 200
    for name, size in outputs_size:
        truth_block = truth[:, start:start + size]
        block = scores[:, start:start + size]
        top2 = _top2(block)
        rows = np.arange(len(block))
        best = (-1.0, 0.0)
        for threshold in grid:
            pred = np.zeros_like(block)
            pred[rows, top2[:, 0]] = 2
            ok = block[rows, top2[:, 1]] >= threshold
            pred[rows[ok], top2[ok, 0]] = 1
            pred[rows[ok], top2[ok, 1]] = 1
            f1 = 2 * np.minimum(pred, truth_block).sum() / (pred.sum() + truth_block.sum())
            best = max(best, (float(f1), float(threshold)))
        thresholds[name] = best[1]
        start += size
    return thresholds

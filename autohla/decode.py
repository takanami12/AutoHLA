"""Bo giai ma lieu mem -> lieu cung. BA bo giai ma khac nhau ton tai trong repo nay
(da tung gay nham lan): `pipelines/export_calls.py` dung top-2 + nguong moi gene do
tren val; `scripts/eval/blend_ridge_pair_cv.py` dung MAP tren cap khong thu tu
(`map_diploid`); `scripts/eval/make_hardcall_pools.py` dung top-2 tran (khong nguong).

`map_diploid` la bo giai ma MAC DINH cua goi nay. `threshold_calls`/`tune_thresholds`
chi ton tai de tai lap dung so cu cua `export_calls.py` (cong C4).
"""
import itertools

import numpy as np
import torch

# Boi so di hop cua Hardy-Weinberg. Mac dinh cua `map_diploid*` va gia tri quay ve
# khi khong fit duoc `tune_tau` -- mot dinh nghia duy nhat de hai cho khong troi.
HW_TAU = 2.0


def map_diploid_pairs(prob: np.ndarray, topk: int = 5, tau: float = HW_TAU):
    """MAP tren cap KHONG THU TU (i, j), i<=j, tren top-k allele theo prob.

    `tau` la BOI SO DI HOP: P(di hop) = tau*p_i*p_j, nen di hop thang dong hop khi
    `tau*p_2 > p_1`. tau=2.0 la he so Hardy-Weinberg va la MAC DINH -- goi khong
    truyen tau cho ra ket qua y het ban goc, tung bit.

    Ly do mo no ra: 2.0 dung khi `prob` la xac suat allele DA HIEU CHUAN, con dau
    ra cua goi nay la sigmoid BCE roi chuan hoa bang `to_prob`, khong phai vay. Do
    tren o VN1K->KHV: mo hinh goi dong hop 16,0% so o trong khi su that 10,5%, va
    o bin `<1%` ty le p_that/p_max trung binh la 0,487 -- nam ngay duoi vach
    p_1/2. Quet tau tren chinh o do: F1 micro 0,8348 (tau=2) -> 0,8543 (tau~5,7),
    bin `<1%` 0,470 -> 0,550, va KHONG bin nao te di. Xem `tune_tau`.

    P(dong hop i) = p_i^2, P(di hop i,j) = 2 p_i p_j (giong homozygous_call trong
    AEHLA/objects/trainer.py). Tra (chi_so_cap, posterior): chi_so_cap la (N, 2)
    int, posterior la (N,) float trong [0, 1] -- xac suat cua CAP DA CHON, dung
    duoc vi prob moi hang da chuan hoa tong 1 (to_prob) nen tong xac suat tren toan
    bo khong gian cap dung bang 1.

    Chep nguyen van tu scripts/eval/blend_ridge_pair_cv.py:70-85 (`map_diploid`),
    tach rieng phan toi da hoa; `map_diploid` ben duoi goi lai ham nay nen hai ham
    khong bao gio bat dong (xem test_pairs_and_dosage_agree).
    """
    n = len(prob)
    pairs = np.zeros((n, 2), dtype=int)
    posterior = np.zeros(n, dtype=float)
    log_tau = np.log(tau)
    for i in range(n):
        top = np.argsort(-prob[i])[:topk]
        logp = np.log(np.maximum(prob[i][top], 1e-12))
        best, arg = -np.inf, (top[0], top[0])
        for u, v in itertools.combinations_with_replacement(range(len(top)), 2):
            value = logp[u] + logp[v] + (log_tau if u != v else 0.0)
            if value > best:
                best, arg = value, (top[u], top[v])
        pairs[i] = arg
        posterior[i] = np.exp(best)
    return pairs, posterior


def map_diploid(prob: np.ndarray, topk: int = 5, tau: float = HW_TAU) -> np.ndarray:
    """MAP tren cap khong thu tu -> ma tran lieu 0/1/2. Bo giai ma MAC DINH.

    Chu ky GIU Y NGUYEN ban goc (scripts/eval/blend_ridge_pair_cv.py::map_diploid)
    de cong C4 so duoc: cung mot dau vao ra cung mot dau ra, xem
    test_matches_the_blend_script_on_random_input -- `tau` la tham so THEM VAO co
    mac dinh 2.0, khong doi duong mac dinh.
    """
    pairs, _ = map_diploid_pairs(prob, topk, tau)
    out = np.zeros_like(prob)
    rows = np.arange(len(prob))
    out[rows, pairs[:, 0]] += 1
    out[rows, pairs[:, 1]] += 1
    return out


def to_prob(matrix: np.ndarray) -> np.ndarray:
    """Dosage/score khong am -> phan bo xac suat moi hang tong 1.

    Chep nguyen van tu scripts/eval/blend_ridge_pair_cv.py:66-68.
    """
    m = np.clip(matrix, 0.0, None) + 1e-6
    return m / m.sum(1, keepdims=True)


def _top2(block) -> np.ndarray:
    """Hai chi so diem cao nhat moi hang -- CHEP NGUYEN VAN bieu thuc torch cua
    AEHLA `block.argsort(dim=1)[:, -2:].flip(1)` (export_calls.py:80,224).

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

    CHI de tai lap so cu cua export_calls.py (cong C4) -- KHONG phai bo giai ma
    mac dinh cua goi nay (dung `map_diploid`). `scores` la ma tran sigmoid da noi
    het cac gene (N, tong_allele); `thresholds` la {ten_gene: nguong} tu
    `tune_thresholds`; `outputs_size` la [(ten_gene, kich_thuoc), ...] cung thu tu
    voi cac khoi trong `scores` (vd `AutoNet.outputs_size`).

    Port cua pipelines/export_calls.py:222-227 (nhanh khong-haprec), chuyen tu
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
    export_calls.py dung de sinh calls.csv. `truth` la dosage THAT 0/1/2 (khong
    phai nhan multi-hot BCE) -- xem `label_dosage`-style constructor o noi goi.

    Port cua pipelines/export_calls.py::tune_thresholds:65-83, chuyen tu torch
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


# tau=2 (Hardy-Weinberg) o giua luoi, va luoi dung log-deu vi tac dong cua tau la
# nhan len ty le p_1/p_2. Tran 32: tren nguong do bo giai ma da gan nhu top-2 cung
# va bin `>=20%` bat dau mat dong hop THAT (do duoc: 0,916 -> 0,881 tu tau 6 -> 32).
# Lam tron: tau di thang vao manifest.json va dong log, va exp/log tra ve
# 7.999999999999998 thay vi 8.0.
TAU_GRID = tuple(np.round(np.exp(np.linspace(np.log(1.0), np.log(32.0), 21)), 3))


def tune_tau(blocks, grid=TAU_GRID, topk: int = 5) -> float:
    """Boi so di hop toi da hoa F1 micro GOP tren `blocks` = [(prob, truth), ...].

    `prob` la phan bo da chuan hoa cua mot gene (mau x allele), `truth` la lieu
    THAT 0/1/2 cung hinh dang. Goi tren du doan OUT-OF-FOLD, khong bao gio tren
    train: mo hinh nen thuoc long train (memory `base-memorizes-train-split`) nen
    fit o do se chon tau = 2 vi moi thu da dung san.

    Hoa thi lay tau NHO NHAT -- ham muc tieu phang tren mot dai rong (do duoc:
    tau 4,8..22,6 chenh nhau 0,004 F1), va tau nho nhat la can thiep it nhat so
    voi hang so Hardy-Weinberg.
    """
    best = (-1.0, float("inf"))
    for tau in grid:
        tp = pred_sum = truth_sum = 0.0
        for prob, truth in blocks:
            pred = map_diploid(prob, topk, tau)
            tp += np.minimum(pred, truth).sum()
            pred_sum += pred.sum()
            truth_sum += truth.sum()
        f1 = 2 * tp / max(pred_sum + truth_sum, 1e-12)
        if f1 > best[0]:
            best = (f1, float(tau))
    return best[1]

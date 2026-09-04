"""Dau doc ridge tren bieu dien `z` cua AutoNet.

Port cua dissect_rare_pathway. Ly do ton tai (do
duoc, khong phai gia dinh): ridge thang fc3 o allele hiem chu yeu vi HAM MAT MAT
(binh phuong thay vi softmax canh tranh), 63% hieu ung, p=0.016 -- do noi bo.

CANH BAO khi dung: loi ich cua tang nay TAT khi du lieu lon. Tren VN1K (851 mau)
no cong +0.055 F1 o bin `<1%`; tren HAN (8.967 mau) beta roi ve 0 o 70/70 o.
Xem do noi bo. Giu lai vi cohort nho la truong hop that,
nhung dung ky vong no cong gi khi panel lon.
"""
import numpy as np


def fit_ridge(z_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    """Ridge da dau ra; dang doi ngau khi p > n, dang chinh khi p <= n.

    Hai dang cho CUNG nghiem, chi khac chi phi: doi ngau giai he n x n, chinh
    giai he p x p. `z` cua AutoNet rong 256 nen cohort duoi 256 mau di duong
    doi ngau.
    """
    n, p = z_train.shape
    if p > n:
        k = z_train @ z_train.T
        alpha = np.linalg.solve(k + lam * np.eye(n), y_train)
        return z_train.T @ alpha
    g = z_train.T @ z_train
    return np.linalg.solve(g + lam * np.eye(p), z_train.T @ y_train)


def apply_ridge(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    return z @ w


LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def zscore(train: np.ndarray, *others: np.ndarray):
    """Chuan hoa theo cot bang trung binh/do lech cua TRAIN (khong phai cua chinh
    tung khoi) -- port dissect_rare_pathway.zscore."""
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(a - mu) / sd for a in (train,) + others]


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC bang thu hang (Mann-Whitney); nan khi mot lop vang mat."""
    positive = labels > 0
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Trung binh thu hang trong moi nhom bang nhau -- khong thi diem hoa lam AUC lech.
    values = scores[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def select_lambda(z_train, y_train, z_val, y_val, lambdas=LAMBDAS):
    """Chon lam bang AUC trung binh tren VAL (port probe_stage), tra (w, lam).

    Dau vao phai da zscore theo train. Bo qua allele khong co ca hai lop tren val
    -- AUC cua no khong dinh nghia duoc, khong phai bang 0.
    """
    best = (-np.inf, None, None)
    for lam in lambdas:
        w = fit_ridge(z_train, y_train, lam)
        scores = apply_ridge(z_val, w)
        aucs = np.array([_auc(scores[:, a], y_val[:, a])
                         for a in range(y_train.shape[1])])
        mean = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else -np.inf
        if mean > best[0]:
            best = (mean, w, lam)
    return best[1], best[2]

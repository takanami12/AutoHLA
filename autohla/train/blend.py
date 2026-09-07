"""Tron log-tuyen tinh arm nen voi arm phu, mot beta moi (gene, bang AF).

    s = normalize( exp[(1-b)*log p_nen + b*log p_phu] )   roi goi cung MAP-diploid.

Beta la MOT vo huong moi (gene, bang AF) nen fit tren du doan OUT-OF-FOLD la
sach (dung luong 1-2 tham so). KHONG duoc fit tren train: mo hinh nen thuoc long
train (top-2 recall 1.000 tren train, do noi bo), nen
moi tang hau nghiem hoc tren train se chon beta = 0.

Tach beta theo AF la BAT BUOC chu khong phai tuy chon: do noi bo
do duoc rang mot beta chung keo bin hiem xuong; nguong mac dinh af_split=0.20 la
gia tri da sinh ra do noi bo

Port cua blend_ridge_pair_cv.
"""
import numpy as np

from ..decode import map_diploid, to_prob

# 21 diem khi mot beta; 11x11 khi tach theo AF (luoi tho vi phai quet 9 fold LOFO).
BETA_GRID = tuple(np.round(np.linspace(0, 1, 21), 2))
COARSE = tuple(np.round(np.linspace(0, 1, 11), 2))


def micro_f1(pred: np.ndarray, truth: np.ndarray, mask=None) -> float:
    """F1 micro tren ma tran lieu cung (mau x allele). `mask` chon tap allele."""
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    t, p = truth.sum(), pred.sum()
    tp = np.minimum(pred, truth).sum()
    sn, ppv = tp / max(t, 1), tp / max(p, 1)
    return 0.0 if sn + ppv == 0 else 2 * sn * ppv / (sn + ppv)


def blend(base_prob, extra_prob, beta_rare, beta_common, freq, *, af_split=0.20):
    """Tron hai phan bo xac suat (mau x allele) da chuan hoa bang `to_prob`.

    `af_split=None` thi dung MOT beta (`beta_rare`) cho moi allele. Dau vao phai
    la XAC SUAT da chuan hoa -- khong tu goi `to_prob` o day vi goi hai lan cong
    1e-6 hai lan va lam lech so so voi ban goc.
    """
    weight = (beta_rare if af_split is None
              else np.where(np.asarray(freq) < af_split, beta_rare, beta_common)[None, :])
    log_base = np.log(np.maximum(base_prob, 1e-12))
    log_extra = np.log(np.maximum(extra_prob, 1e-12))
    return to_prob(np.exp((1 - weight) * log_base + weight * log_extra))


def _grid(af_split, grid):
    if grid is not None:
        return tuple(grid)
    return ([(b, b) for b in BETA_GRID] if af_split is None
            else [(r, c) for r in COARSE for c in COARSE])


def _choose(entries, grid, af_split, decode=map_diploid, band=None):
    """Chon (beta_rare, beta_common) toi da hoa F1 TRUNG BINH tren cac khoi trong
    `entries` -- trung binh theo khoi, KHONG gop chung roi tinh mot F1 (hai cach
    cho so khac nhau; ban goc trung binh theo fold).

    Moi phan tu cua `entries` la (prob_nen, prob_phu, truth, freq).
    """
    # Cong don bang float Python roi chia, KHONG dung np.mean: np.mean cong theo
    # cap nen lam tron khac, va khi hai o cua luoi HOA (xay ra that:
    # LPHASE7 fold 2 DQA1, ca hai ra 0.984949) thu tu lam tron quyet dinh argmax
    # chon o nao. Do la khac biet duy nhat giua bang beta cua goi va bang cong bo.
    scores = []
    for r, c in grid:
        total = 0.0
        for pb, pe, truth, freq in entries:
            total += micro_f1(decode(blend(pb, pe, r, c, freq, af_split=af_split)),
                              truth, band)
        scores.append(total / len(entries))
    best = int(np.argmax(scores))
    return grid[best][0], grid[best][1], float(scores[best])


def fit_beta(base_oof, extra_oof, truth, freq, *, af_split=0.20, grid=None):
    """Mot beta moi (gene, bang AF). Fit tren du doan OUT-OF-FOLD.

    Bon dau vao la dict theo gene: `base_oof[gene]`/`extra_oof[gene]`/`truth[gene]`
    la (N, A), `freq[gene]` la (A,). Tra {gene: (beta_rare, beta_common)}.
    """
    grid = _grid(af_split, grid)
    return {gene: _choose([(base_oof[gene], extra_oof[gene], truth[gene],
                            np.asarray(freq[gene]))], grid, af_split)[:2]
            for gene in base_oof}

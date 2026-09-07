"""Tron log-tuyen tinh arm nen voi arm phu, mot beta moi (gene, bang AF).

    s = normalize( exp[(1-b)*log p_nen + b*log p_phu] )   roi goi cung MAP-diploid.

Beta la MOT vo huong moi (gene, bang AF) nen fit tren du doan OUT-OF-FOLD la
sach (dung luong 1-2 tham so). KHONG duoc fit tren train: mo hinh nen thuoc long
train (top-2 recall 1.000 tren train, memory `base-memorizes-train-split`), nen
moi tang hau nghiem hoc tren train se chon beta = 0.

Tach beta theo AF la BAT BUOC chu khong phai tuy chon: `ridge-blend-readout-passes`
do duoc rang mot beta chung keo bin hiem xuong; nguong mac dinh af_split=0.20 la
gia tri da sinh ra results/rare_pathway/beta_AEHLA_PAIR_RIDGE_AF.csv.

Port cua scripts/eval/blend_ridge_pair_cv.py.
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


# ---------------------------------------------------------------------------
# Che do tuong thich (§6): nap dung 10 fold CV cu roi tra bang beta 70 dong.
# Duong dan duoi day la cua REPO NAY -- ham nay chi de tai lap so cu, duong di
# cong khai la fit_beta() tren split noi bo cua goi (train/splits.py).
# ---------------------------------------------------------------------------
LOCI = ("A", "B", "C", "DPB1", "DRB1", "DQA1", "DQB1")
_COMPAT_LABELS = {
    "VN1K": "data/label/DGV4VN_1015.HISAT_result.resolution.4digits.csv",
    "1KGP_VN1K": "data/label/1KGP_VN1K.HLA.resolution.4digits.csv",
    "HAN": "HAN_dataset/HAN.HLA.4digit.tsv",
}


def _wide(frame, gene, samples, alleles):
    """(mau x allele) dosage, thieu thi 0."""
    block = frame[frame.gene.eq(gene)]
    return (block.pivot_table(index="sample_id", columns="allele", values="dosage",
                              aggfunc="sum")
            .reindex(index=samples, columns=alleles).fillna(0.0).to_numpy(float))


def _truth_matrix(labels, samples, gene, alleles):
    index = {a: i for i, a in enumerate(alleles)}
    out = np.zeros((len(samples), len(alleles)))
    for row, sample in enumerate(samples):
        for copy in (1, 2):
            value = labels.at[sample, f"{gene}_{copy}"]
            if isinstance(value, str) and value in index:
                out[row, index[value]] += 1
    return out


def load_cv_cache(pred_dir, base: str, extra: str, cohort: str, protocol,
                  folds=range(1, 11)):
    """Nap du doan CV cu -> ({(fold, gene): (prob_nen, prob_phu, truth, af)},
    {gene: allele universe}, {fold: sample list}).

    Tach rieng khoi `blend_cv_dir` vi tools/compare_flows.py::beta_source can
    CHINH cache nay nhung gan lai fold theo split noi bo cua goi.
    """
    from pathlib import Path

    import pandas as pd

    from ..io.labels import load_labels

    pred_dir, protocol = Path(pred_dir), Path(protocol)
    label_path = Path(_COMPAT_LABELS.get(cohort, cohort))
    if not label_path.is_absolute():
        label_path = Path(__file__).resolve().parents[3] / label_path
    labels = load_labels(str(label_path), list(LOCI), n_digits=4)

    def load(name):
        frame = pd.read_csv(pred_dir / f"{name}.dosage.csv.gz")
        # ID mau HAN la so nen pandas doc thanh int con nhan la chuoi -- ep ca hai
        # phia ve chuoi thay vi de no im lang ra F1 = 0.
        frame["sample_id"] = frame.sample_id.astype(str)
        return frame

    def fold_freq(fold, gene, alleles):
        """AF train cua fold, theo gene -- dinh nghia CLAUDE.md 2026-08-18."""
        train = (protocol / f"fold{fold:02d}/train.samples").read_text().split()
        values = pd.Series(labels.loc[[s for s in train if s in labels.index],
                                      [f"{gene}_1", f"{gene}_2"]].to_numpy().ravel())
        values = values[values.notna() & values.ne("")]
        counts = values.value_counts()
        total = max(len(values), 1)
        return np.array([counts.get(a, 0) / total for a in alleles])

    base_frame, extra_frame = load(base), load(extra)
    universe = {g: sorted(set(base_frame[base_frame.gene.eq(g)].allele)
                          | set(extra_frame[extra_frame.gene.eq(g)].allele))
                for g in LOCI}
    cache, fold_samples = {}, {}
    for fold in folds:
        b = base_frame[base_frame.fold.eq(fold)]
        e = extra_frame[extra_frame.fold.eq(fold)]
        samples = fold_samples[fold] = sorted(set(b.sample_id) & set(e.sample_id))
        for gene in LOCI:
            alleles = universe[gene]
            cache[(fold, gene)] = (to_prob(_wide(b, gene, samples, alleles)),
                                   to_prob(_wide(e, gene, samples, alleles)),
                                   _truth_matrix(labels, samples, gene, alleles),
                                   fold_freq(fold, gene, alleles))
    return cache, universe, fold_samples


def fit_beta_from_cv_dir(pred_dir, base: str, extra: str, cohort: str, protocol,
                         af_split: float = 0.20, folds=range(1, 11)):
    """CHE DO TUONG THICH: nap 10 fold CV cu tu `pred_dir` + `protocol` roi chon
    beta leave-one-fold-out cho tung (fold, gene).

    Tra [{fold, gene, beta, beta_common, lofo_f1}] -- cung schema voi
    results/rare_pathway/beta_*.csv de so truc tiep.
    """
    return blend_cv_dir(pred_dir, base, extra, cohort, protocol, af_split, folds)[0]


def blend_cv_dir(pred_dir, base: str, extra: str, cohort: str, protocol,
                 af_split: float = 0.20, folds=range(1, 11)):
    """Nhu tren nhung tra CA hai: (bang beta, frame goi cung da tron).

    Mot luot duy nhat -- luoi 11x11 x 9 fold LOFO x 7 gene la phan dat nhat cua
    ca tang nay, khong chay hai lan de lay hai ket qua. `cohort` la ten cohort da
    biet hoac duong dan toi file nhan.
    """
    import pandas as pd

    cache, universe, fold_samples = load_cv_cache(pred_dir, base, extra, cohort,
                                                  protocol, folds)
    grid = _grid(af_split, None)
    rows, calls = [], []
    for fold in folds:
        for gene in LOCI:
            others = [cache[(o, gene)] for o in folds if o != fold]
            beta, beta_common, lofo_f1 = _choose(others, grid, af_split)
            rows.append(dict(fold=fold, gene=gene, beta=beta,
                             beta_common=beta_common, lofo_f1=lofo_f1))
            pb, pe, _, freq = cache[(fold, gene)]
            call = map_diploid(blend(pb, pe, beta, beta_common, freq,
                                     af_split=af_split))
            for i, sample in enumerate(fold_samples[fold]):
                for a in np.where(call[i] > 1e-4)[0]:
                    calls.append((sample, fold, gene, universe[gene][a],
                                  int(call[i, a])))
    frame = pd.DataFrame(calls,
                         columns=["sample_id", "fold", "gene", "allele", "dosage"])
    return rows, frame

"""Split k-fold noi bo, cat tu chinh VCF train.

Thay giao thuc CV co san cua repo (do noi bo) de goi
chay duoc tren du lieu la. Chia theo MAU chu khong theo hang haplotype: mot mau
la hai hang trong VCF, tach doi chung se ro ri hoan toan.
"""
import numpy as np


def kfold_by_sample(sample_ids, k: int = 10, seed: int = 77):
    """Tra [(train, test), ...] gom k cap, moi mau nam trong DUNG mot test fold.

    Tat dinh theo `seed`. Mac dinh k=10, seed=77 -- giao thuc co dinh cua repo
    (CLAUDE.md), de so sanh voi cac bang cu duoc.
    """
    ids = sorted(sample_ids)
    if k < 2 or k > len(ids):
        raise ValueError("k phai trong [2, {}], nhan duoc {}".format(len(ids), k))
    order = np.random.default_rng(seed).permutation(len(ids))
    folds = []
    for i in range(k):
        test = sorted(ids[j] for j in order[i::k])
        folds.append((sorted(set(ids) - set(test)), test))
    return folds

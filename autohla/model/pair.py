"""Tang cham diem CAP khong thu tu (diploid) cong tren logit cua AutoNet.

Port cua run_pair_energy_cv, BO ba nhanh da bi
bac bo bang so do (khong phai bo cho gon):
  - `PAIR_ENERGY_MLP` (Arm C): hai co y nghia o MOI AF bin, ppv sut trong khi sn
    giu -- qua khop. Xem do noi bo.
  - `pair_prior` / `PAIR_PRIOR_ALPHA` (Arm A): am o moi AF bin; unary da chua tan
    suat roi nen cong log-prior cap la dem hai lan. Xem do noi bo.
  - SWA (Arm B): cong 5-10% -0.014, mot fold phan ky lam trung binh trong so no.
    Xem do noi bo.

Giu nguyen `nn.init.zeros_` tren `context`: no lam diem xuat phat cua tang nay
DUNG BANG diem ban goc da chuan hoa, tuong tac chi hoc sau khi gradient chay.
"""
import math

import torch
from torch import nn


class PairEnergyHead(nn.Module):
    """Phan phoi cap khong thu tu voi tuong tac hang thap dung chung."""

    def __init__(self, hidden_dim: int, sizes, rank: int = 16,
                 use_interaction: bool = True):
        super__init__
        self.sizes = tuple(int(size) for size in sizes)
        self.use_interaction = bool(use_interaction)
        # Thu tu tao tham so (context truoc, embeddings sau) la mot phan cua hop
        # dong: doi cho lam lech dong RNG so voi ban goc va test doi chieu do.
        self.context = nn.Linear(hidden_dim, rank)
        self.allele_embeddings = nn.ParameterList
        for gene, size in enumerate(self.sizes):
            i, j = torch.triu_indices(size, size)
            lookup = torch.full((size, size), -1, dtype=torch.long)
            pair_id = torch.arange(len(i), dtype=torch.long)
            lookup[i, j] = pair_id
            lookup[j, i] = pair_id
            self.register_buffer(f"pair_i_{gene}", i, persistent=False)
            self.register_buffer(f"pair_j_{gene}", j, persistent=False)
            self.register_buffer(f"pair_lookup_{gene}", lookup, persistent=False)
            self.allele_embeddings.append(nn.Parameter(torch.randn(size, rank) *.1))
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

    def pair_indices(self, gene: int, device):
        return (getattr(self, f"pair_i_{gene}").to(device),
                getattr(self, f"pair_j_{gene}").to(device))

    def forward(self, shared, score):
        """`shared` (N, hidden_dim) tu AutoNet.encode; `score` (N, tong_allele) la
        sigmoid da noi het cac gene. Tra (logits, prob, dosage) theo tung gene."""
        context = self.context(shared)
        logits, probabilities, dosage = [], [], []
        start = 0
        for gene, size in enumerate(self.sizes):
            block = score[:, start:start + size]
            unary = torch.logit(block.clamp(1e-6, 1 - 1e-6))
            i, j = self.pair_indices(gene, score.device)
            if self.use_interaction:
                interaction = (context[:, None,:]
                               * self.allele_embeddings[gene][i]
                               * self.allele_embeddings[gene][j]).sum(-1)
            else:
                interaction = score.new_zeros((score.shape[0], len(i)))
            logits_g = unary[:, i] + unary[:, j]
            # log 2 cho cap di hop: (a,b) va (b,a) la cung mot kieu gen.
            logits_g = logits_g + (i != j).to(score.dtype) * score.new_tensor(math.log(2))
            logits_g = logits_g + interaction
            prob_g = torch.softmax(logits_g, dim=1)
            dosage_g = score.new_zeros((score.shape[0], size))
            dosage_g.scatter_add_(1, i[None].expand(score.shape[0], -1), prob_g)
            dosage_g.scatter_add_(1, j[None].expand(score.shape[0], -1), prob_g)
            logits.append(logits_g)
            probabilities.append(prob_g)
            dosage.append(dosage_g)
            start += size
        return logits, probabilities, dosage

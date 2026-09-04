"""AutoNet: port cua AENet's LIVE PATH (champion config) only.

Doc AENet (__init__, make_layers),:883-1060 (dense_input,
bottleneck),:1017-1100 (split_dosage, lphase_forward),:1152-1235 (forward, readout)
truoc khi doc file nay.

Thu tu KHOI TAO MODULE trong __init__ theo dung thu tu make_layers cho nhanh song:
CoNetBackbone (encoder1 -> encoder2 -> decoder1 -> decoder2 -> output_norm)
-> ReconstructionHead -> Linear(shared) -> HLA_Blocks moi gene. Thu tu nay quyet dinh
thu tu tieu thu RNG, ma cong C2 so tung tensor mot -- sai thu tu la lech gia tri
backbone ngay tu buoc khoi tao, du gia tri cuoi cung khong doi (S1 chi luu backbone +
reconstruction_head, duoc khoi tao TRUOC ca shared lan HLA_Blocks trong chu ky nay).

`phased` gop AE_LPHASE=1 + AE_LPHASE_ORACLE=1 cua AENet lam MOT tham so constructor:
san xuat luon dung pha THAT (Beagle trong fold) lam oracle, khong bao gio dung
Phaser hoc duoc (da dong, xem memory "phase precision cliff") nen lop do bi bo
hoan toan, khong con trong file nay. S1 luon huan luyen KHONG pha (xem train/pretrain.py).
"""
import torch
import torch.nn as nn

from autohla.model.backbone import CoNetBackbone
from autohla.model.heads import HLA_Blocks, ReconstructionHead

_DIM = 64
_HEAD_DROPOUT = 0.1
_MASK_DROPOUT = 0.05


class AutoNet(nn.Module):
    def __init__(self, input_size, outputs_size, group, *, phased=False,
                shared_dim=256, strides=(2, 2), device=None, head="full"):
        super__init__
        self.input_size = input_size
        self.outputs_size = outputs_size
        self.group = group
        self.phased = phased
        self.device = device
        self.gene_names = [name for name, _ in outputs_size]
        # Trunk chi con hai kenh (dosage, missing): kenh cM do duoc la null tren
        # F1 o moi AF bin, va lop stem gop duoc vao encoder1 nen ca hai da bi go.
        # Khong cong them kenh cho phased: hang hap1 cua dau vao phased bi
        # dense_input tach ra TRUOC khi toi backbone (xem duoi) nen khong lam rong
        # kenh trunk -- dung cong thuc in_channels cua AENet.make_layers.
        self._missing_row = 1
        self.backbone = CoNetBackbone(dim=_DIM, strides=strides).to(device)
        self.reconstruction_head = ReconstructionHead(_DIM).to(device)

        bottleneck_length = -(-input_size // self.backbone.total_stride)
        self.shared = nn.Sequential(
            nn.Linear(_DIM * bottleneck_length, shared_dim), nn.LayerNorm(shared_dim),
            nn.GELU, nn.Dropout(_HEAD_DROPOUT),
        ).to(device)
        self.head = head
        self.HLA_Blocks = nn.ModuleDict
        for name, output_size in outputs_size:
            self.HLA_Blocks[name] = HLA_Blocks(name, shared_dim, output_size, device,
                                               lean=head == "lean")

    # ---- S1 pretext (train/pretrain.py goi truc tiep) -----------------------

    def dense_input(self, x):
        """(B, 3|4, L) -> ((B, 2, L) [dosage, missing], hap1 hoac None).

        Hang hap1 (khi phased) luon la hang CUOI -- xem autohla.io.dataset.load_dataset.
        """
        x = x.reshape(-1, x.shape[-2], self.input_size)
        hap1 = None
        if self.phased:
            if x.shape[1] != 4:
                raise ValueError(
                    "phased=True needs a 4-channel input (OR, AND, missing, "
                    "hap1), got {}".format(x.shape[1]))
            hap1 = x[:, -1]
            x = x[:,:-1]
        dosage = x[:, 0] + x[:, 1]
        missing = x[:, 2]
        return torch.stack([dosage, missing], dim=1), hap1

    def corrupt(self, dense):
        """Che ngau nhien cho pretext S1 (AENet.corrupt; dynamic_rows luon la (0,)
        o cau hinh vo dich vi khong co AE_STEM=dom)."""
        missing_row = self._missing_row
        masked = (torch.rand_like(dense[:, 0]) < _MASK_DROPOUT) & (dense[:, missing_row] == 0)
        corrupted = dense.clone
        corrupted[:, 0][masked] = 0
        corrupted[:, missing_row][masked] = 1
        return corrupted, masked

    def recon_forward(self, dense):
        u, _, _, raw_length = self.backbone(dense)
        return self.reconstruction_head(u, raw_length)

    # ---- S2 readout -----------------------------------------------------------

    def bottleneck(self, dense):
        padded, valid, _ = self.backbone.pad(dense)
        return self.backbone.encode(padded, valid)

    def split_dosage(self, dense, pi):
        """(dense, pi) -> (dense1, dense2): 2 ban sao haplotype tren thang dosage.
        g1 + g2 == 2g dung theo CAU TRUC (xem AENet.split_dosage)."""
        g = dense[:, 0]
        delta = (g == 1).to(g.dtype) * (2 * pi - 1)
        dense1, dense2 = dense.clone, dense.clone
        dense1[:, 0] = g + delta
        dense2[:, 0] = g - delta
        return dense1, dense2

    def _shared_of(self, dense):
        return self.shared(self.bottleneck(dense).flatten(1))

    def readout(self, outs):
        return torch.cat(outs, dim=1)

    def trunk_readout(self, dense):
        shared = self._shared_of(dense)
        if not self.gene_names:
            return shared.new_zeros(shared.shape[0], 0)
        return self.readout([self.HLA_Blocks[name](shared) for name in self.gene_names])

    def lphase_forward(self, dense, hap1):
        """Hai luot trunk tren hai haplotype tiem an, gop bang noisy-OR
        (AENet.lphase_forward): p = 1 - (1-p1)(1-p2)."""
        pi = hap1.to(dense.dtype)
        dense1, dense2 = self.split_dosage(dense, pi)
        p1, p2 = self.trunk_readout(dense1), self.trunk_readout(dense2)
        return (1 - (1 - p1) * (1 - p2)).clamp(1e-7, 1 - 1e-7)

    def forward(self, x):
        dense, hap1 = self.dense_input(x)
        if self.phased:
            return self.lphase_forward(dense, hap1)
        return self.trunk_readout(dense)

    def encode(self, x):
        """z dung chung cho pair/ridge: shared embedding (N, shared_dim), TRUOC HLA_Blocks."""
        dense, hap1 = self.dense_input(x)
        if self.phased:
            pi = hap1.to(dense.dtype)
            dense1, dense2 = self.split_dosage(dense, pi)
            return 0.5 * (self._shared_of(dense1) + self._shared_of(dense2))
        return self._shared_of(dense)

    # ---- checkpoint (S1 chi luu backbone + reconstruction_head) ---------------

    def save_s1(self, path):
        torch.save({"backbone": self.backbone.state_dict,
                    "reconstruction_head": self.reconstruction_head.state_dict}, path)

    def load_s1(self, path):
        state = torch.load(path, map_location=torch.device(self.device or "cpu"))
        self.backbone.load_state_dict(state["backbone"])
        self.reconstruction_head.load_state_dict(state["reconstruction_head"])

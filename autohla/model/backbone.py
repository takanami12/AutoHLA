"""Trunk: port cua AENet (ConvBlock, CoNetBackbone).

Chi giu nhanh SONG cua make_layers voi cau hinh vo dich (champion): bins=False,
tf_l1=0, dec_attn=False, stem_mode='' luon dung -- nen CmCrossAttention, bin_cuts,
tf_l1/tf_gate va stem_mlp (chi phuc vu cac nhanh do) bi bo hoan toan, khong con la
tham so. `interpolate_valid`/`pad`/`encode1`/`encode`/`forward` giu dung logic goc
cho phan con lai.

Trunk nhan DUNG hai kenh: dosage, missing. Kenh cM (ca doc gia lan cM that noi
suy tu ban do di truyen) va lop stem da bi go han: cM do duoc la null tren F1 o
moi AF bin, con stem la mot phep tuyen tinh dat truoc mot phep tuyen tinh khac
nen gop vao encoder1 la tuong duong ve bieu dien ma bo duoc 96,9% tham so lop
dau (262.464 -> 12.352).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel, stride=1, in_channels=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels or channels, channels, kernel, stride,
                      (kernel - stride) // 2),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CoNetBackbone(nn.Module):
    """dim=64, strides=(2,2) o cau hinh vo dich. in_channels LA 2 (dosage,
    missing); pad() dinh vi hang missing theo dung hang so do."""

    def __init__(self, dim=64, strides=(4, 4)):
        super().__init__()
        if dim % 8:
            raise ValueError("dim must be divisible by 8")
        if any(stride % 2 for stride in strides):
            # ConvBlock dem (kernel - stride) // 2 voi kernel chan, stride le se lam
            # mat mot vi tri va z het khop length // total_stride.
            raise ValueError("encoder strides must be even")
        self.strides = tuple(strides)
        self.total_stride = strides[0] * strides[1]
        self.encoder1 = ConvBlock(dim, 64, strides[0], in_channels=2)
        self.encoder2 = ConvBlock(dim, 64, strides[1])
        self.decoder1 = ConvBlock(dim, 63)
        self.decoder2 = ConvBlock(dim, 63)
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def interpolate_valid(valid, size):
        return F.interpolate(
            valid.float().unsqueeze(1), size=size, mode="nearest").squeeze(1).bool()

    def pad(self, x):
        raw_length = x.shape[-1]
        padding = (-raw_length) % self.total_stride
        valid = torch.ones(x.shape[0], raw_length, dtype=torch.bool, device=x.device)
        if not padding:
            return x, valid, raw_length
        rows = [F.pad(x[:, 0], (0, padding), value=0),                    # dosage
                F.pad(x[:, 1], (0, padding), value=1)]                    # missing
        rows += [F.pad(x[:, i], (0, padding), value=0)
                for i in range(2, x.shape[1])]
        return torch.stack(rows, 1), F.pad(valid, (0, padding), value=False), raw_length

    def encode1(self, x):
        return self.encoder1(x)

    def encode(self, x, valid):
        """encoder1 -> encoder2. `valid` giu trong chu ky goi de doi khop voi AENet
        (tf_l1 dung no lam mask attention; o day khong con nhanh do nua)."""
        return self.encoder2(self.encode1(x))

    def forward(self, x):
        x, valid, raw_length = self.pad(x)
        length = x.shape[-1]
        level1_length = length // self.strides[0]
        z = self.encode(x, valid)
        decoded1 = self.decoder1(F.interpolate(z, size=level1_length, mode="nearest"))
        h = self.decoder2(F.interpolate(decoded1, size=length, mode="nearest"))
        return self.output_norm(h.transpose(1, 2)).transpose(1, 2), z, valid, raw_length

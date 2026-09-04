"""Dau ra: port cua nnet HLA_Blocks + AENet ReconstructionHead.

`_HEAD_CONFIG` chep tu model.json, chi giu 7 gene trong GROUPS cua
AutoHLA (bo HLA_DPA1, khong dung trong 4 group duoc ho tro). `self.relu`/`self.elu`
cua ban goc bi bo: ELU khong bao gio duoc goi trong forward goc (chet tu dau), va
ReLU khong co tham so nen doi sang goi F.relu khong doi RNG/state_dict.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_HEAD_CONFIG = {
    "HLA_A":    {"fc1_len": 256, "fc2_len": 128, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_B":    {"fc1_len": 384, "fc2_len": 256, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_C":    {"fc1_len": 256, "fc2_len": 128, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_DRB1": {"fc1_len": 384, "fc2_len": 256, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_DQA1": {"fc1_len": 384, "fc2_len": 256, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_DQB1": {"fc1_len": 256, "fc2_len": 128, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
    "HLA_DPB1": {"fc1_len": 384, "fc2_len": 256, "p_dropout_1": 0.3, "p_dropout_2": 0.1},
}


class HLA_Blocks(nn.Module):
    """`lean=True` bo fc1/fc2 (cung bn/dropout di kem): fc3 doc THANG `z`.

    Ly do: ridge theo chang o `<1%` cho z 0.6814 vs h 0.6448 (do noi bo) -- `h`, tuc dau ra fc1/fc2 huan luyen duoi BCE, da vut
    tin hieu allele hiem TRUOC khi fc3 nhin thay. Cho fc3 doc `z` la dua no ve
    dau vao tot hon 0.037.

    Khac biet KHONG chi la it tham so: head lean cung khong con BatchNorm va
    Dropout, nen no thay mot phan bo dau vao khac han chu khong phai cung mot
    phan bo qua it lop hon.
    """

    def __init__(self, name, input_size, output_size, device=None, lean=False):
        super__init__
        self.lean = bool(lean)
        if self.lean:
            self.fc3 = nn.Linear(input_size, output_size).to(device)
            return
        cfg = _HEAD_CONFIG[name]
        fc1_len, fc2_len = cfg["fc1_len"], cfg["fc2_len"]
        self.fc1 = nn.Linear(input_size, fc1_len).to(device)
        self.bn1 = nn.BatchNorm1d(fc1_len).to(device)
        self.fc2 = nn.Linear(fc1_len, fc2_len).to(device)
        self.bn2 = nn.BatchNorm1d(fc2_len).to(device)
        self.fc3 = nn.Linear(fc2_len, output_size).to(device)
        self.dropout1 = nn.Dropout(p=cfg["p_dropout_1"])
        self.dropout2 = nn.Dropout(p=cfg["p_dropout_2"])

    def forward(self, x):
        if self.lean:
            return torch.sigmoid(self.fc3(x))
        out = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        out = self.dropout2(F.relu(self.bn2(self.fc2(out))))
        return torch.sigmoid(self.fc3(out))


class ReconstructionHead(nn.Module):
    """S1: bottleneck -> logits tai tao dosage {0,1,2} tren luoi chip."""

    def __init__(self, dim):
        super__init__
        self.projection = nn.Conv1d(dim, 3, 1)

    def forward(self, u, raw_length):
        return self.projection(u)[:,:,:raw_length]

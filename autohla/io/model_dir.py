"""Hop dong thu muc `model/` (spec §4.1).

    model/
      manifest.json   # phien ban, RunConfig da dung, n_train, phased_rate, he so tron
      markers.tsv     # marker CHOT LAI luc train
      encoder.pkl     # allele universe
      trunk.pt  head.pt  [pair.pt]  [ridge.npz]

Tach trunk/head lam hai file la CO Y: dau doc la thu duy nhat gan voi mot allele
universe cu the, con trunk thi khong. Tach ra thi doc mot mo hinh de biet no thay
gi (`trunk.pt` chung) va no goi ra gi (`head.pt`) khong can nap ca hai.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 2
_HEAD_PREFIX = "HLA_Blocks."


def _split_state(state):
    trunk = {k: v for k, v in state.items() if not k.startswith(_HEAD_PREFIX)}
    head = {k: v for k, v in state.items() if k.startswith(_HEAD_PREFIX)}
    return trunk, head


def save_model(out_dir, net, cfg, markers, encoder, beta=None, pair=None,
               ridge=None, af_split=0.20) -> None:
    """`beta` la {gene: (beta_rare, beta_common)}; `pair`/`ridge` la state_dict cua
    PairEnergyHead va ma tran trong so ridge, ca ba deu tuy chon.

    `af_split` phai di CUNG beta: mot cap (beta_rare, beta_common) khong co nghia
    neu khong biet nguong nao chia hai bang."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trunk, head = _split_state(net.state_dict())
    torch.save(trunk, out / "trunk.pt")
    torch.save(head, out / "head.pt")
    if pair is not None:
        torch.save(pair, out / "pair.pt")
    if ridge is not None:
        np.savez(out / "ridge.npz", **{g: w for g, w in ridge.items()})

    with open(out / "markers.tsv", "w") as fh:
        for marker in markers:
            fh.write("\t".join(marker) + "\n")
    with open(out / "encoder.pkl", "wb") as fh:
        pickle.dump(encoder, fh)

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(cfg).items()},
        "n_markers": len(markers),
        "input_size": int(net.input_size),
        "outputs_size": [[name, int(size)] for name, size in net.outputs_size],
        "head": getattr(net, "head", "full"),
        "beta": None if beta is None else {g: list(v) for g, v in beta.items()},
        "af_split": float(af_split),
        "has_pair": pair is not None,
        "has_ridge": ridge is not None,
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _check_format(manifest) -> None:
    """v1 (truoc khi go kenh cM va stem) VAN doc duoc -- nhung chi khi no ghi
    dung cau hinh ma goi nay con dung duoc: cm_channel=False, cm_decay=False,
    stem=False. Ba co do la arm `cm_off_nostem`, kien truc trung khop tuyet doi
    voi ban rut gon, nen 40 o CV cu nap lai va chay hau ky duoc khong can train
    lai. v1 nao khac ba co do co trunk 3 kenh hoac co lop stem: tu choi o day,
    kem ly do, thay vi de load_state_dict no ra 'size mismatch'."""
    version = manifest["format_version"]
    if version == FORMAT_VERSION:
        return
    if version != 1:
        raise ValueError("model/ ghi boi phien ban {} nhung goi nay doc phien ban {}"
                         .format(version, FORMAT_VERSION))
    legacy = {k: bool(manifest.get(k, True)) for k in ("cm_channel", "cm_decay", "stem")}
    if any(legacy.values()):
        raise ValueError(
            "model/ v1 nay dung {} -- ban hien tai da go kenh cM va lop stem nen "
            "khong nap duoc. Chi model v1 co cm_channel=False, cm_decay=False, "
            "stem=False moi tuong thich.".format(
                ", ".join(sorted(k for k, v in legacy.items() if v))))


def load_model(model_dir) -> dict:
    """Tra {'manifest', 'markers', 'encoder', 'net', 'beta', 'pair', 'ridge'}.

    Dung lai AutoNet tu manifest roi nap trunk+head -- khong doan hinh dang tu
    checkpoint, vi doan sai thi nap duoc nhung sai gene.
    """
    from ..model.net import AutoNet
    from .vcf import read_markers

    path = Path(model_dir)
    with open(path / "manifest.json") as fh:
        manifest = json.load(fh)
    _check_format(manifest)
    cfg = manifest["config"]
    net = AutoNet(manifest["input_size"],
                  [[name, size] for name, size in manifest["outputs_size"]],
                  cfg["group"], phased=cfg["phased"], shared_dim=cfg["shared_dim"],
                  strides=tuple(cfg["strides"]),
                  head=manifest.get("head", "full"))
    state = torch.load(path / "trunk.pt", map_location="cpu")
    state.update(torch.load(path / "head.pt", map_location="cpu"))
    net.load_state_dict(state)
    net.eval()

    with open(path / "encoder.pkl", "rb") as fh:
        encoder = pickle.load(fh)
    ridge = None
    if manifest["has_ridge"]:
        with np.load(path / "ridge.npz") as data:
            ridge = {g: data[g] for g in data.files}
    return {
        "manifest": manifest,
        "markers": read_markers(str(path / "markers.tsv")),
        "encoder": encoder,
        "net": net,
        "beta": (None if manifest["beta"] is None
                 else {g: tuple(v) for g, v in manifest["beta"].items()}),
        "pair": (torch.load(path / "pair.pt", map_location="cpu")
                 if manifest["has_pair"] else None),
        "ridge": ridge,
    }

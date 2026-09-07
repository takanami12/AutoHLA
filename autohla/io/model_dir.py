"""Hop dong thu muc `model/` (spec §4.1).

    model/
      manifest.json   # phien ban, RunConfig da dung, n_train, phased_rate, he so tron
      markers.tsv     # marker CHOT LAI luc train
      encoder.json    # train-only allele universe (no pickle execution)
      trunk.pt  head.pt  [pair.pt]  [ridge.npz]

Tach trunk/head lam hai file la CO Y: dau doc la thu duy nhat gan voi mot allele
universe cu the, con trunk thi khong. Tach ra thi doc mot mo hinh de biet no thay
gi (`trunk.pt` chung) va no goi ra gi (`head.pt`) khong can nap ca hai.
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from ..decode import HW_TAU
from ..model.pair import PAIR_RANK

FORMAT_VERSION = 3
_HEAD_PREFIX = "HLA_Blocks."


def _split_state(state):
    trunk = {k: v for k, v in state.items() if not k.startswith(_HEAD_PREFIX)}
    head = {k: v for k, v in state.items() if k.startswith(_HEAD_PREFIX)}
    return trunk, head


def save_model(out_dir, net, cfg, markers, encoder, beta=None, pair=None,
               pair_net=None, ridge=None, af_split=0.20, tau=HW_TAU,
               pair_rank=PAIR_RANK, train_af=None, training_provenance=None) -> None:
    """`beta` la {gene: (beta_rare, beta_common)}; `pair`/`ridge` la state_dict cua
    PairEnergyHead va ma tran trong so ridge, ca ba deu tuy chon.

    `af_split` phai di CUNG beta: mot cap (beta_rare, beta_common) khong co nghia
    neu khong biet nguong nao chia hai bang.

    `tau` la boi so di hop cua bo giai ma (xem decode.tune_tau). No thuoc ve model/
    chu khong phai ve lenh impute: no duoc fit tren du doan out-of-fold cua CHINH
    mo hinh nay, nen dung lai cho mo hinh khac la vo nghia."""
    # Nua tang pair la rac IM LANG: ridge duoc fit tren z cua trunk NEN, con arm
    # nen doc trunk da fine-tune. Thieu mot trong hai thi luc impute hai tang
    # dang doc hai mo hinh khac nhau ma khong gi bao loi.
    if (pair is None) != (pair_net is None):
        raise ValueError("pair and pair_net must be given together (got pair="
                         "{}, pair_net={})".format(pair is not None,
                                                   pair_net is not None))
    if beta is not None and af_split is not None and train_af is None:
        raise ValueError("AF-split blend requires train_af")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trunk, head = _split_state(net.state_dict())
    torch.save(trunk, out / "trunk.pt")
    torch.save(head, out / "head.pt")
    if pair is not None:
        torch.save(pair, out / "pair.pt")
        pair_trunk, pair_head = _split_state(pair_net.state_dict())
        torch.save(pair_trunk, out / "pair_trunk.pt")
        torch.save(pair_head, out / "pair_head.pt")
    if ridge is not None:
        np.savez(out / "ridge.npz", **{g: w for g, w in ridge.items()})

    with open(out / "markers.tsv", "w") as fh:
        for marker in markers:
            fh.write("\t".join(marker) + "\n")
    alleles = {gene: [encoder.decoder[(gene, i)] for i in range(size)]
               for gene, size in encoder.label_counter.items()}
    with open(out / "encoder.json", "w") as fh:
        json.dump(alleles, fh, sort_keys=True)

    manifest = {
        "format_version": FORMAT_VERSION,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in vars(cfg).items()},
        "n_markers": len(markers),
        "input_size": int(net.input_size),
        "outputs_size": [[name, int(size)] for name, size in net.outputs_size],
        "head": getattr(net, "head", "full"),
        "beta": None if beta is None else {g: list(v) for g, v in beta.items()},
        "af_split": None if af_split is None else float(af_split),
        "tau": float(tau),
        "has_pair": pair is not None,
        "pair_rank": int(pair_rank),
        "has_ridge": ridge is not None,
        "train_af": None if train_af is None else {g: np.asarray(v).tolist()
                                                   for g, v in train_af.items()},
        "training_provenance": training_provenance,
        "decoder": "marginal",
    }
    files = ["trunk.pt", "head.pt", "markers.tsv", "encoder.json"]
    if pair is not None:
        files += ["pair.pt", "pair_trunk.pt", "pair_head.pt"]
    if ridge is not None:
        files += ["ridge.npz"]
    manifest["sha256"] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                           for name in files}
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _build_net(manifest):
    from ..model.net import AutoNet
    cfg = manifest["config"]
    return AutoNet(manifest["input_size"],
                   [[name, size] for name, size in manifest["outputs_size"]],
                   cfg["group"], phased=cfg["phased"], shared_dim=cfg["shared_dim"],
                   strides=tuple(cfg["strides"]),
                   head=manifest.get("head", "full"))


def _check_format(manifest) -> None:
    """v1 (truoc khi go kenh cM va stem) VAN doc duoc -- nhung chi khi no ghi
    dung cau hinh ma goi nay con dung duoc: cm_channel=False, cm_decay=False,
    stem=False. Ba co do la arm `cm_off_nostem`, kien truc trung khop tuyet doi
    voi ban rut gon, nen 40 o CV cu nap lai va chay hau ky duoc khong can train
    lai. v1 nao khac ba co do co trunk 3 kenh hoac co lop stem: tu choi o day,
    kem ly do, thay vi de load_state_dict no ra 'size mismatch'."""
    version = manifest["format_version"]
    if version in (2, FORMAT_VERSION):
        return
    if version != 1:
        raise ValueError("model/ was written by format version {} but this "
                         "package reads version {}"
                         .format(version, FORMAT_VERSION))
    legacy = {k: bool(manifest.get(k, True)) for k in ("cm_channel", "cm_decay", "stem")}
    if any(legacy.values()):
        raise ValueError(
            "this v1 model/ uses {} -- the cM channel and the stem layer have "
            "been removed, so it cannot be loaded. Only v1 models with "
            "cm_channel=False, cm_decay=False, stem=False are compatible.".format(
                ", ".join(sorted(k for k, v in legacy.items() if v))))


def load_model(model_dir) -> dict:
    """Tra {'manifest', 'markers', 'encoder', 'net', 'beta', 'pair', 'ridge'}.

    Dung lai AutoNet tu manifest roi nap trunk+head -- khong doan hinh dang tu
    checkpoint, vi doan sai thi nap duoc nhung sai gene.
    """
    from ..model.pair import PairEnergyHead
    from .vcf import read_markers

    if TorchVersion(torch.__version__) < "2.10.0":
        raise ValueError("model loading requires torch>=2.10.0 (CVE-2026-24747)")
    path = Path(model_dir)
    with open(path / "manifest.json") as fh:
        manifest = json.load(fh)
    _check_format(manifest)
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("legacy model uses an unsafe pickle encoder; retrain as format v3")
    if manifest.get("decoder") != "marginal":
        raise ValueError("unsupported model decoder")
    required = {"trunk.pt", "head.pt", "markers.tsv", "encoder.json"}
    if manifest["has_pair"]:
        required.update(("pair.pt", "pair_trunk.pt", "pair_head.pt"))
    if manifest["has_ridge"]:
        required.add("ridge.npz")
    if set(manifest.get("sha256", {})) != required:
        raise ValueError("model checksums must cover exactly all required artifacts")
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"model artifact checksum mismatch: {name}")
    if manifest.get("beta") is not None and manifest.get("af_split") is not None:
        if not manifest.get("train_af"):
            raise ValueError("legacy AF-split model lacks train_af; retrain before impute")
    cfg = manifest["config"]
    net = _build_net(manifest)
    state = torch.load(path / "trunk.pt", map_location="cpu", weights_only=True)
    state.update(torch.load(path / "head.pt", map_location="cpu", weights_only=True))
    net.load_state_dict(state)
    net.eval()

    from .dataset import _Encoder
    encoder = _Encoder()
    with open(path / "encoder.json") as fh:
        alleles = json.load(fh)
    for name, size in manifest["outputs_size"]:
        gene = name.replace("HLA_", "")
        values = alleles[gene]
        if (len(values) != size or len(set(values)) != size
                or any(not isinstance(v, str) or not v.strip() for v in values)):
            raise ValueError(f"encoder does not match output classes for {gene}")
        encoder.label_counter[gene] = size
        for i, value in enumerate(values):
            encoder.decoder[(gene, i)] = value
            encoder.encoder[(gene, value)] = np.eye(size)[i]
        if manifest.get("train_af") is not None:
            af = np.asarray(manifest["train_af"].get(gene), dtype=float)
            if (af.shape != (size,) or not np.isfinite(af).all() or (af < 0).any()
                    or not np.isclose(af.sum(), 1)):
                raise ValueError(f"invalid frozen train_af for {gene}")
    ridge = None
    if manifest["has_ridge"]:
        with np.load(path / "ridge.npz") as data:
            ridge = {g: data[g] for g in data.files}
    pair = None
    if manifest["has_pair"]:
        missing = [n for n in ("pair_trunk.pt", "pair_head.pt", "pair.pt")
                   if not (path / n).exists()]
        if missing:
            raise ValueError("manifest says has_pair but {} is missing from {}"
                             .format(", ".join(missing), path))
        pair_net = _build_net(manifest)
        state = torch.load(path / "pair_trunk.pt", map_location="cpu", weights_only=True)
        state.update(torch.load(path / "pair_head.pt", map_location="cpu", weights_only=True))
        pair_net.load_state_dict(state)
        pair_net.eval()
        head = PairEnergyHead(cfg["shared_dim"],
                              [size for _, size in manifest["outputs_size"]],
                              rank=manifest.get("pair_rank", PAIR_RANK))
        head.load_state_dict(torch.load(path / "pair.pt", map_location="cpu", weights_only=True))
        head.eval()
        pair = {"net": pair_net, "head": head}
    return {
        "manifest": manifest,
        "markers": read_markers(str(path / "markers.tsv")),
        "encoder": encoder,
        "net": net,
        "beta": (None if manifest["beta"] is None
                 else {g: tuple(v) for g, v in manifest["beta"].items()}),
        "pair": pair,
        "ridge": ridge,
        "train_af": manifest.get("train_af"),
    }

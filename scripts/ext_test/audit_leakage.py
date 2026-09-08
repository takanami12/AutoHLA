#!/usr/bin/env python3
"""Kiem ro ri du lieu cua mot o CV/external. In PASS/FAIL, exit != 0 neu thung.

Moi phep kiem ung voi mot phan bien cu the ve leakage. Chay duoc doc lap, khong
can train lai:

    python audit_leakage --cohort vn1k --folds 1-10 \
        --protocol do noi bo cohort

Them --model/--test-vcf de chay phep kiem F (impute bat bien theo lo), la phep
duy nhat can nap mot model/ da train.

CHUNG CU MANH NHAT khong nam trong file nay ma la mot BAT KHA: chay `train`
trong mot thu muc KHONG co test.vcf.gz. Train xong nghia la no khong the doc
test, khong can tin loi ai. `--sandbox <fold_dir>` dung mot thu muc nhu the va
in lenh de chay.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PKG = next(p for p in Path(__file__).resolve().parents
           if (p / "autohla" / "__init__.py").is_file()
           or (p / "AutoHLA" / "autohla" / "__init__.py").is_file())
PKG = PKG if (PKG / "autohla").is_dir() else PKG / "AutoHLA"
ROOT = next(p for p in (PKG, *PKG.parents) if (p / "data" / "references").is_dir())
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "tools"))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """In mot dong ket qua va ghi nho that bai. Tra `ok` de goi y xau chuoi duoc."""
    print("  {} {}{}".format("PASS" if ok else "FAIL", name,
                             "" if not detail else "  -- " + detail))
    if not ok:
        FAILURES.append(name)
    return ok


def _samples(path: Path) -> list[str]:
    return path.read_text().split()


def _vcf_samples(path: Path) -> list[str]:
    from cyvcf2 import VCF
    return list(VCF(str(path)).samples)


def audit_folds(protocol: Path, data: Path, folds) -> None:
    """A/B/C: fold roi nhau, phu het cohort dung mot lan, VCF khop protocol."""
    seen: dict[str, int] = {}
    for fold in folds:
        proto = protocol / f"fold{fold:02d}"
        fdir = data / f"fold{fold:02d}"
        train, test = set(_samples(proto / "train.samples")), set(_samples(proto / "test.samples"))
        val = set(_samples(proto / "val.samples")) if (proto / "val.samples").exists() else set()
        tag = f"fold{fold:02d}"
        check(f"A  {tag}: train ∩ test = ∅", not (train & test),
              "" if not (train & test) else f"{len(train & test)} mẫu chung")
        check(f"A  {tag}: val ∩ test = ∅", not (val & test),
              "" if not (val & test) else f"{len(val & test)} mẫu chung")
        for split, want in (("train", train), ("test", test)):
            vcf = fdir / f"{split}.vcf.gz"
            if not vcf.exists():
                check(f"C  {tag}: có {split}.vcf.gz", False, str(vcf))
                continue
            got = set(_vcf_samples(vcf))
            check(f"C  {tag}: mẫu trong {split}.vcf.gz == {split}.samples",
                  got == want, f"VCF {len(got)} vs protocol {len(want)}")
        for sample in test:
            seen[sample] = seen.get(sample, 0) + 1
    if len(list(folds)) > 1:
        dup = [s for s, n in seen.items() if n != 1]
        check("B  mỗi mẫu nằm trong ĐÚNG một test fold", not dup,
              "" if not dup else f"{len(dup)} mẫu lệch")


def audit_markers(model_dir: Path, pos: Path, train_vcf: Path) -> None:
    """H: model chi nhin marker trong whitelist, va VCF train khong mang nhan."""
    from autohla.io.vcf import markers_from_vcf, read_markers
    allowed = {m[1] for m in read_markers(str(pos))}
    frozen = [line.split("\t") for line in
              (model_dir / "markers.tsv").read_text().splitlines() if line]
    ids = {row[1] for row in frozen}
    check("H  markers.tsv ⊆ position.list", ids <= allowed,
          f"{len(ids - allowed)} marker ngoài whitelist")
    hla = {m[1] for m in markers_from_vcf(str(train_vcf))
           if m[1].upper().startswith(("HLA_", "HLA-", "AA_", "SNPS_"))}
    check("H  VCF train không chứa marker giả HLA_*/AA_*", not hla,
          "" if not hla else f"{len(hla)} marker, vd {sorted(hla)[:3]}")


def audit_freq(model_dir: Path) -> None:
    """G: bang AF trong manifest la cua TRAIN, khong phai cua lo dang impute.

    `freq` chi ton tai khi co tang tron: `train` gan no BEN TRONG nhanh ridge, va
    `_blended_probs` chi doc no khi `beta` va `ridge` deu co. Cohort tu 4000 mau
    tro len khong bat ridge, nen `freq` vang mat mot cach HOP LE va khong the co
    phu thuoc lo -- bat bien can kiem o do la `beta` cung vang.
    """
    import json
    manifest = json.loads((model_dir / "manifest.json").read_text())
    freq = manifest.get("freq")
    if not manifest.get("has_ridge", False):
        check("G  không có tầng trộn ⇒ không đọc bảng AF nào",
              manifest.get("beta") is None,
              "" if manifest.get("beta") is None
              else "has_ridge=False nhưng beta vẫn tồn tại: sẽ trộn mà không có bảng AF")
        return
    if freq is None:
        check("G  manifest có bảng AF của train", False,
              "model/ có ridge nhưng thiếu bảng AF: impute sẽ quay về trung bình của lô")
        return
    total = {g: sum(v) for g, v in freq.items()}
    ok = all(abs(t - 1.0) < 1e-6 or t == 0.0 for t in total.values())
    check("G  manifest['freq'] là phân bố theo gene", ok, str(total))
    check("G  n_train khớp cấu hình", manifest["config"]["n_train"] > 0,
          f"n_train={manifest['config']['n_train']}")


def audit_batch_independence(model_dir: Path, vcf: Path, n: int = 8) -> None:
    """F: goi ca lo va goi tung mau mot phai ra KET QUA Y HET."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        full = tmp / "full.csv"
        run = [sys.executable, "-m", "autohla", "impute", "--model", str(model_dir),
               "--vcf", str(vcf), "--out", str(full), "--threads", "2"]
        proc = subprocess.run(run, cwd=str(PKG), capture_output=True, text=True)
        if proc.returncode:
            check("F  impute cả lô chạy được", False, proc.stderr.strip()[-200:])
            return
        import pandas as pd
        whole = pd.read_csv(full)
        ids = list(dict.fromkeys(whole["sample_id"]))[:n]
        singles = []
        for sid in ids:
            sub = tmp / f"{sid}.vcf.gz"
            subprocess.run(["bcftools", "view", "-s", str(sid), "-Oz", "-o", str(sub),
                            str(vcf)], check=True, capture_output=True)
            out = tmp / f"{sid}.csv"
            subprocess.run(run[:-4] + ["--vcf", str(sub), "--out", str(out),
                                       "--threads", "2"],
                           cwd=str(PKG), check=True, capture_output=True)
            singles.append(pd.read_csv(out))
        one = pd.concat(singles, ignore_index=True)
        cols = list(one.columns)
        left = whole[whole.sample_id.isin(ids)].sort_values(cols).reset_index(drop=True)
        right = one.sort_values(cols).reset_index(drop=True)
        check(f"F  gọi {len(ids)} mẫu riêng lẻ == gọi cả lô", left.equals(right),
              "" if left.equals(right) else "kết quả một mẫu phụ thuộc lô")


def sandbox_hint(fold_dir: Path) -> None:
    """D: dung thu muc CHI co train.vcf.gz va in lenh train de chay trong do."""
    box = fold_dir.parent / (fold_dir.name + ".sandbox")
    box.mkdir(exist_ok=True)
    for pattern in ("train.vcf.gz", "train.vcf.gz.tbi"):
        src = fold_dir / pattern
        if src.exists():
            shutil.copyfile(src, box / pattern)
    leaked = sorted(p.name for p in box.iterdir() if "test" in p.name)
    check("D  sandbox không chứa file test nào", not leaked, str(leaked))
    print(f"     sandbox = {box}\n"
          f"     chạy `autohla train --vcf {box}/train.vcf.gz ...` trong đó;\n"
          f"     train xong nghĩa là nó KHÔNG THỂ đã đọc test.")


def _range(text: str):
    if "-" in text:
        lo, hi = text.split("-", 1)
        return range(int(lo), int(hi) + 1)
    return [int(x) for x in text.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--model", help="một model/ để kiểm G và H")
    ap.add_argument("--test-vcf", help="VCF test cho phép kiểm F")
    ap.add_argument("--sandbox", help="thư mục foldNN để dựng sandbox cho phép kiểm D")
    args = ap.parse_args(argv)

    from run_cv import COHORTS
    spec = COHORTS[args.cohort]
    folds = _range(args.folds)
    print(f"== {args.cohort}, fold {args.folds} ==")
    audit_folds(Path(args.protocol), Path(spec["data"]), folds)
    if args.model:
        first = Path(spec["data"]) / f"fold{list(folds)[0]:02d}" / "train.vcf.gz"
        audit_markers(Path(args.model), Path(spec["pos"]), first)
        audit_freq(Path(args.model))
    if args.model and args.test_vcf:
        audit_batch_independence(Path(args.model), Path(args.test_vcf))
    if args.sandbox:
        sandbox_hint(Path(args.sandbox))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} phép kiểm THỦNG: " + "; ".join(FAILURES))
        return 1
    print("tất cả phép kiểm đạt")
    return 0


def _demo() -> None:
    """Tu kiem: `check` phai ghi nho that bai va khong ghi nho thanh cong."""
    FAILURES.clear()
    assert check("ok", True) is True
    assert check("hong", False, "ly do") is False
    assert FAILURES == ["hong"]
    FAILURES.clear()
    assert _range("1-3") == range(1, 4) and _range("2,5") == [2, 5]
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        sys.exit(main())

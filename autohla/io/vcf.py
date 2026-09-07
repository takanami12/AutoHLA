"""Doc VCF va bang marker.

`load_haplotypes` la port NGUYEN VAN cua AEHLA/src/data_helper.py:81-190
(`load_vcf_file`) -- khong doi thu tu vong lap, dtype, hay reshape, vi cong C1
so tensor nay bit-for-bit voi ben AEHLA. Khong doc os.environ o day.
"""
import warnings

import numpy as np
import pandas as pd
from cyvcf2 import VCF

GROUPS: dict[int, list[str]] = {
    1: ["A"],
    2: ["B", "C"],
    3: ["DPB1"],
    4: ["DRB1", "DQA1", "DQB1"],
}

# Vung nhiem sac the cho tung group, chep tu AEHLA/configs/references/hla_regions.json
# (chi 4 group AutoHLA ho tro). AEHLA dung no de loc bang marker chung (ca vung MHC)
# xuong cua so cua tung group ben trong load_ref_positions; load_haplotypes lam
# dung viec do o duoi day.
_REGIONS: dict[int, dict] = {
    1: {"CHROM": "chr6", "START": 29725988, "END": 31169169},
    2: {"CHROM": "chr6", "START": 31069169, "END": 31657158},
    3: {"CHROM": "chr6", "START": 32986042, "END": 33480577},
    4: {"CHROM": "chr6", "START": 32223340, "END": 32829113},
}


def read_markers(path: str) -> list[tuple[str, str, str, str]]:
    """Doc file position list -> [(CHROM, POS, REF, ALT)].

    Port phan doc-file cua AEHLA load_ref_positions (data_helper.py:24-35), tru
    loc theo group -- loc do chuyen vao load_haplotypes vi no can `group`.
    """
    lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            lines.append(line.strip())
    lines = sorted(set(lines))
    markers = []
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 4, (
            "Reference position file is not in tab-delimited format: "
            "CHROM POS REF ALT")
        markers.append(tuple(parts))
    return markers


def markers_from_vcf(path: str) -> list[tuple[str, str, str, str]]:
    """Danh sach marker suy tu CHINH file VCF, cung dang voi `read_markers`.

    Dung khi nguoi dung khong truyen --marker-list: "chip" khi do la dung nhung
    marker file train co. Chi lay bien the hai allele (mot ALT) -- phan con lai
    load_haplotypes cung khong doc duoc.
    """
    markers = set()
    for variant in VCF(path):
        if len(variant.ALT) != 1:
            continue
        markers.add((str(variant.CHROM), str(variant.POS), str(variant.REF),
                     str(variant.ALT[0])))
    return sorted(markers)

def scan_vcf(path: str) -> dict:
    """{'n_samples', 'phased_rate', 'sample_ids'}.

    Khong nap genotype vao RAM -- chi duyet toi da 200 bien the dau de uoc
    phased_rate tu variant.genotypes[:, 2].
    """
    vcf = VCF(path)
    sample_ids = list(vcf.samples)
    n_called = n_phased = 0
    for i, variant in enumerate(vcf):
        if i >= 200:
            break
        genotypes = np.asarray(variant.genotypes, dtype=np.int8)
        if genotypes.shape[1] <= 2:
            continue
        called = (genotypes[:, 0] >= 0) & (genotypes[:, 1] >= 0)
        n_called += int(called.sum())
        n_phased += int((genotypes[called, 2] != 0).sum())
    return {
        "n_samples": len(sample_ids),
        "phased_rate": n_phased / max(n_called, 1),
        "sample_ids": sample_ids,
    }


def load_haplotypes(path: str, markers, group: int, absent_value: int = -1,
                    require_phased: bool = False) -> pd.DataFrame:
    """Port NGUYEN VAN AEHLA/src/data_helper.py load_vcf_file (dong 81-190).

    Doi tham so ref_pos_path -> markers (da doc san qua read_markers) va bo
    nt_channels; phan con lai KHONG doi mot dong logic nao. Index la
    '<sample>_1'/'<sample>_2' xen ke theo dung thu tu mau, cot la marker.

    require_phased: fail if the VCF is not actually phased. The two rows per
        sample are read from variant.genotypes[:, 0] and [:, 1] whatever the
        separator is, so an unphased file loads WITHOUT error and silently
        supplies a meaningless row 1 -- writers normalise unphased hets to 0/1,
        which pins hap1 to the REF allele at every het.
    """
    if ".vcf" not in path:
        raise ValueError("Input file must be a vcf.gz file")

    n_rows = 0

    start_pos, end_pos = _REGIONS[group]["START"], _REGIONS[group]["END"]
    ref_position = [list(x) for x in markers
                    if int(x[1]) >= int(start_pos) and int(x[1]) <= int(end_pos)]

    vcf = VCF(path)
    samples = np.array(vcf.samples)

    try:
        requested_chrom = str(_REGIONS[group]["CHROM"])
        # HAN la hg18 va dat ten nhiem sac the 6 la ``6``; VCF VN1K dung ``chr6``.
        # Khop mot trong hai cach viet ma khong doi danh sach marker hay toa do.
        chrom = next((name for name in vcf.seqnames
                      if name == requested_chrom
                      or name.removeprefix("chr") == requested_chrom.removeprefix("chr")),
                     requested_chrom)
        variant_pos_range = "{}:{}-{}".format(chrom, start_pos, end_pos)
        _vcf = vcf(variant_pos_range)
        if _vcf != None:
            vcf = _vcf
    except Exception:
        raise ValueError("vcf file must be indexed before using range query")

    sample_list_1 = [x + "_1" for x in samples]
    sample_list_2 = [x + "_2" for x in samples]
    data = np.full((len(samples), 2, len(ref_position)), absent_value, dtype=np.int8)
    n_called = n_phased = 0

    # absent_value dien vao cac marker chip ma vcf nay khong co. Ben goi dung
    # missing channel truyen -1, cung trung voi -1 cyvcf2 tra cho genotype ./.,
    # nen ca hai loai "thieu" ra khoi day cung mot gia tri sentinel.
    selected_ref_pos = {}
    for _ref_pos in ref_position:
        selected_ref_pos[" ".join(_ref_pos)] = 1
    pos_dict = {}

    for i, pos in enumerate(ref_position):
        pos_dict[" ".join(pos)] = i

    for variant in vcf:
        key = (str(variant.CHROM) + " " + str(variant.POS) + " " + variant.REF
               + " " + variant.ALT[0])
        if key not in selected_ref_pos:
            continue
        if selected_ref_pos[key] > 1:
            warnings.warn("Duplicate position: {}".format(key))
            continue
        selected_ref_pos[key] += 1
        genotypes = np.asarray(variant.genotypes, dtype=np.int8)
        pos_index = pos_dict[key]
        data[:, 0, pos_index] = genotypes[:, 0]
        data[:, 1, pos_index] = genotypes[:, 1]
        if require_phased and genotypes.shape[1] > 2:
            called = (genotypes[:, 0] >= 0) & (genotypes[:, 1] >= 0)
            n_called += int(called.sum())
            n_phased += int((genotypes[called, 2] != 0).sum())
        n_rows += 1

        if n_rows == len(selected_ref_pos):
            break

    headers = ["_".join(x) for x in ref_position]

    if require_phased:
        phased_rate = n_phased / max(n_called, 1)
        if phased_rate < 0.95:
            raise ValueError(
                "Phase-dependent load asked for {} but only {:.1%} of called "
                "genotypes carry the phased flag. Row 1 would not be a "
                "haplotype. Phase the file first, or call with "
                "require_phased=False.".format(path, phased_rate))

    overlap_rate = n_rows / len(ref_position)
    if overlap_rate < 1.0:
        warnings.warn("Overlap position ratio is {}".format(overlap_rate))
    if overlap_rate < 0.5:
        raise ValueError(
            "Overlap position ratio is too low: {}, ensure that all "
            "microarray markers are highly overlapped in vcf file"
            .format(overlap_rate))

    # reshape() nha hang theo THU TU MAU (s0_1, s0_2, s1_1, s1_2, ...), dung
    # bang thu tu chen dict; nhan phai xen ke y het, khong phai noi hai danh
    # sach -- noi la gan haplotype cua mau nay cho ten mau khac.
    df = pd.DataFrame(data.reshape(len(samples) * 2, len(ref_position)),
                      index=[name for pair in zip(sample_list_1, sample_list_2)
                             for name in pair],
                      columns=headers)
    return df

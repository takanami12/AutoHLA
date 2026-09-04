"""Bo chon cau hinh. Doc tu chinh VCF."""
from dataclasses import dataclass

from autohla.io.vcf import scan_vcf

# Nguong pha cua data_helper. Duoi nguong nay hang hap1 khong
# phai haplotype, chi la genotype da chuan hoa -> nhanh phased vo nghia.
PHASED_MIN_RATE = 0.95
# Heuristic TIET KIEM COMPUTE, khong phai phat hien. Chi co hai co mau da do
# (851 va 8.967); 4000 la diem giua. An toan nam o cho he so tron tu ve 0 khi
# tang do vo dung, nen vuot nguong hay khong deu khong ra ket qua sai.
PAIR_SKIP_N = 4000


@dataclass(frozen=True)
class RunConfig:
    group: int
    phased: bool
    n_train: int
    phased_rate: float
    use_pair: bool
    use_ridge: bool
    markers_path: str | None = None
    phase_skip_loci: tuple[str, ...] = ("DPB1",)
    shared_dim: int = 256
    strides: tuple[int, int] = (2, 2)
    rare_bce_max: float = 10.0
    seed: int = 77
    head: str = "full"

    @classmethod
    def from_vcf(cls, vcf_path, group, *, phase="auto", marker_list=None,
                 force_pair=None, force_ridge=None, head="full"):
        if head not in ("full", "lean"):
            raise ValueError("head phai la full|lean, nhan duoc {!r}".format(head))
        if phase not in ("auto", "on", "off"):
            raise ValueError("phase phai la auto|on|off, nhan duoc {!r}".format(phase))
        info = scan_vcf(vcf_path)
        rate, n = info["phased_rate"], info["n_samples"]
        if phase == "on" and rate < PHASED_MIN_RATE:
            raise ValueError(
                "--phase on nhung chi {:.2f} genotype mang co phased. Hang hap1 se "
                "khong phai haplotype. Pha file truoc, hoac dung --phase off."
                .format(rate))
        # phase="auto" khong con sniff phased_rate de quyet dinh -- do la mot loi
        # trong chinh ban thiet ke: ca ba VCF do duoc deu bao phased_rate=1.0000,
        # vi mot cong cu (imputer/phaser) da ghi dau '|' cho MOI genotype trong
        # file. Co '|' chi noi "mot cong cu co chay qua file nay", khong noi
        # "hang hap1 la haplotype THAT" -- dung canh bao trong chinh docstring
        # cua load_haplotypes. Vi vay auto KHONG THE phan biet mot VCF chua pha
        # that voi mot VCF da pha that -- hai truong hop nay lech +0.161 F1 o
        # bin hiem <1%. Pha la OPT-IN: chi "on" moi bat, "auto" va "off" deu tat.
        phased = phase == "on"
        small = n < PAIR_SKIP_N
        return cls(group=int(group), phased=phased, n_train=n, phased_rate=rate,
                   use_pair=small if force_pair is None else bool(force_pair),
                   use_ridge=small if force_ridge is None else bool(force_ridge),
                   markers_path=marker_list, head=head)

    def explain(self):
        lines = [
            "n_train      = {}".format(self.n_train),
            "phased_rate  = {:.4f}".format(self.phased_rate),
            "phased       = {}  (nguong {})".format(self.phased, PHASED_MIN_RATE),
            "phase_skip   = {}  (DPB1 dao dau: <1% -0.0440)".format(
                ", ".join(self.phase_skip_loci) or "-"),
            "use_pair     = {}  (huan luyen khi n_train < {})".format(
                self.use_pair, PAIR_SKIP_N),
            "use_ridge    = {}  (he so tu ve 0 thi tang bi go khoi model/)".format(
                self.use_ridge),
            "markers      = {}".format(self.markers_path or "suy tu VCF train"),
            "head         = {}  (lean = z -> fc3 thang, -87% tham so dau doc)"
            .format(self.head),
        ]
        return "\n".join(lines)

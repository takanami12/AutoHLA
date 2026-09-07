# AutoHLA

Four-digit HLA allele calling from SNP-chip genotypes in a VCF. Two commands, nothing else.

```bash
autohla train  --vcf train.vcf.gz --labels labels.csv --group 1 --out model/
autohla impute --model model/ --vcf test.vcf.gz --out calls.csv
```

## Install

```bash
git clone https://github.com/takanami12/AutoHLA.git && cd AutoHLA
pip install .              # installs the `autohla` command
```

Requires Python >= 3.10. Dependencies (`torch`, `numpy`, `pandas`, `cyvcf2`) are
pulled in by pip. Without installing, `python -m autohla --help` works from the
repository root.

## `train`

| flag | default | meaning |
|---|---|---|
| `--vcf` | — | genotype VCF of the training set |
| `--labels` | — | label table (see *Label format*) |
| `--group` | — | 1 = A · 2 = B,C · 3 = DPB1 · 4 = DRB1,DQA1,DQB1 |
| `--out` | — | `model/` directory to write |
| `--marker-list` | inferred from the training VCF | chip marker list; the model may only look at markers on this list |
| `--phase` | `auto` | phasing is **opt-in**: only `on` enables it (see *Phasing*) |
| `--epochs` / `--s1-epochs` | 100 | epochs per stage |
| `--no-s1` | off | skip the S1 pretext stage (measured null on downstream F1) |
| `--no-posthoc` | off | skip the ridge + blend layer (several times faster) |
| `--posthoc-folds` | 10 | internal folds used to fit the post-hoc layer |
| `--af-split` | 0.20 | allele-frequency threshold splitting the rare and common blend coefficients |
| `--threads` | 8 | torch threads; left alone, torch takes every core on the machine |

## `impute`

A sample's call does **not** depend on which other samples are imputed with it.
The allele-frequency table that selects the blend coefficient is the one measured on
the training set and stored in `model/manifest.json`; it is never re-estimated from the
batch being called.

| flag | default | meaning |
|---|---|---|
| `--model` | — | `model/` directory written by `train` |
| `--vcf` | — | genotype VCF to call |
| `--out` | — | output `calls.csv` |
| `--threads` | 8 | torch threads |

## Label format

CSV or TSV (the separator is sniffed). First column is the sample ID, then
`<GENE>_1`, `<GENE>_2`:

```csv
sample_id,A_1,A_2,B_1,B_2
NG001,11:01,24:02,15:02,40:01
NG002,A*33:03,A*11:01,B*58:01,B*15:02
```

Both `A*33:03` and `33:03` are accepted — the gene prefix is stripped. This
mismatch once drove F1 to zero silently, so it is handled in exactly one place
rather than left to the caller.

## Phasing

`--phase on` uses the first haplotype row as phased input. **Off by default,
including under `auto`.** Reason: the `|` flag in a VCF only says *some tool
wrote this file*, not that the first row is a real haplotype — all three
cohorts measured during development reported `phased_rate = 1.0000` while only
one of them was truly phased. Turning it on wrongly costs 0.212 F1 in the rare
allele band, so it has to be your choice, not the package's guess. `--phase on`
still checks `phased_rate >= 0.95` and refuses below it.

DPB1 is excluded from the phased branch (`phase_skip`); phasing reverses sign
on that locus.

## Configuration report

`train` prints the configuration it picked before running:

```
n_train      = 851
phased_rate  = 1.0000
phased       = False  (threshold 0.95)
phase_skip   = DPB1  (phasing reverses sign on DPB1: <1% -0.0440)
use_pair     = True  (trained when n_train < 4000)
use_ridge    = True  (dropped from model/ if the coefficient goes to 0)
markers      = inferred from the training VCF
```

The ridge layer and its blend coefficients are only enabled below 4,000 training
samples. On larger cohorts the blend coefficient collapses to zero on its own, so
the package drops the layer instead of paying compute for a zero.

## `model/`

```
model/
  manifest.json    # format version, the RunConfig used, n_train, phased_rate,
                   # blend coefficients, the training-set allele frequencies
                   # (`freq`) and the homozygote multiplier `tau`
  markers.tsv      # markers frozen at training time
  encoder.pkl      # allele universe
  trunk.pt  head.pt  [pair.pt]  [ridge.npz]
```

`encoder.pkl` is a pickle, so loading a `model/` directory executes code from it.
Only load model directories you trained yourself or obtained from a source you
trust.

## License

MIT — see [LICENSE](LICENSE).

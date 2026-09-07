# AutoHLA

Four-digit HLA allele calling from SNP-chip genotypes. The product workflow has
two commands:

```bash
autohla train --vcf train.vcf.gz --labels labels.csv --group 1 \
  --marker-list ref.APMDA.position.list --out model/
autohla impute --model model/ --vcf test.vcf.gz --out calls.csv
```

## Install

From this release directory:

```bash
pip install .
autohla --help
```

Requires Python >= 3.10 and Torch >=2.10.0, NumPy, pandas and cyvcf2. Alternatively,
`python -m autohla --help` works from this directory.

## Inputs

Use compressed, indexed VCFs. Reference and target VCFs must use the same
coordinate build, chromosome names and REF/ALT orientation as the marker list.
The required `--marker-list` is a tab-separated file with four columns:
`CHROM POS REF ALT`. The model only reads those markers, within each gene
group's window, and freezes the list in `model/markers.tsv`.
Imputation stops below 50% marker overlap; this checks coverage, not complete
allele/build harmonization.

Labels are CSV or TSV, with sample ID first and two allele columns per gene:

```csv
sample_id,A_1,A_2,B_1,B_2
NG001,11:01,24:02,15:02,40:01
NG002,A*33:03,A*11:01,B*58:01,B*15:02
```

Gene prefixes are stripped. Training preserves both binary presence targets
for BCE and copy-count truth (0/1/2) for pair training and model selection.
Missing gene labels are not treated as negative alleles.

## Train

| Flag | Default | Meaning |
|---|---|---|
| `--vcf`, `--labels`, `--group`, `--out`, `--marker-list` | required | training data, gene group, output model and chip markers |
| `--val-vcf` | omitted | separate validation VCF; otherwise hold out 5% of training samples |
| `--phase` | `auto` | only `on` enables phased input; `auto` and `off` disable it |
| `--epochs` / `--s1-epochs` | 100 | epoch limits per stage |
| `--no-s1` | off | skip S1 pretraining |
| `--no-posthoc` | off | skip pair, ridge and blend |
| `--posthoc-folds` | 10 | internal folds for out-of-fold blend fitting |
| `--pair` | `on` | pair head: `on`, `off`, or `auto` by training sample count |
| `--pair-epochs` | 40 | pair fine-tuning epoch limit |
| `--af-split` | 0.20 | training AF threshold separating the two blend coefficients |
| `--head` | `full` | `lean` removes the first two readout layers |
| `--strides` | `2,2` | the two encoder strides |
| `--gene-loss-weight` | empty | per-gene weight, for example `DQA1=0.25` |
| `--threads` | 8 | Torch threads; also accepted by impute |

Group 1: A; group 2: B,C; group 3: DPB1; group 4: DRB1,DQA1,DQB1.
See `autohla train --help` for all available options.
The default training seed is 77.

Training runs optional S1 pretraining, 2-digit then 4-digit curriculum,
enabled pair/ridge layers, decoder fitting and model export.
Ridge is enabled below 4,000 training samples. Pair is on by default;
the sample-count rule only applies to `--pair auto`.
`--no-posthoc` also disables pair.

With `--val-vcf`, the supplied validation split is used without another
holdout. Training and validation sample IDs must be disjoint. Allele frequency
is counted from training labels only: allele copies divided by valid HLA
copies of that same gene. Validation and target samples do not enter this
table. The table is saved with the model, so target batch composition cannot
change the blend coefficients.

Phasing is opt-in. `--phase on` requires reliable phased input and at least
95% phased flags among called genotypes. A VCF `|` flag alone does not
establish phasing accuracy. `--phase on` applies to every gene in the selected
group, including DPB1; the default `auto` does not use phasing.

## Impute and model files

```csv
sample_id,gene,allele_1,allele_2,posterior
NG001,A,11:01,24:02,0.8421
```

One row per sample and gene, with two allele copies. Homozygous calls repeat
the allele. `posterior` is the normalized probability of the called unordered
pair under the decoder, bounded by [0,1]. It is not an empirically calibrated
probability of correctness. The default decoder uses pair marginal dosage
when enabled, blends ridge when available, and applies the model's fitted
heterozygote multiplier `tau`.

```text
model/
  manifest.json    # format v3, configuration, beta/tau, train_af, training_provenance
  markers.tsv
  encoder.json
  trunk.pt  head.pt
  pair_trunk.pt  pair_head.pt  pair.pt   # when pair is enabled
  ridge.npz                             # when ridge is enabled
```

Provenance records the training split and data/configuration identifiers.
The manifest checks SHA-256 hashes of model artifacts; weights are loaded with
`weights_only=True` and the allele universe uses JSON. Legacy v1/v2 models using
pickle are rejected and require retraining. Checksums detect changed files;
they do not authenticate the model's supplier.
Only load trusted models. PyTorch <=2.9.1 has a checkpoint-loading vulnerability
even with `weights_only=True`; the patched release is 2.10.0 according to the
[PyTorch advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
This is a research CLI, not an untrusted-upload service or clinically validated
software. Do not publish data, `work/`, or `training_provenance.json` containing
sample IDs. Review the manifest before sharing: provenance and reference AF
are not privacy-protected by a checksum.

## Evaluation status

Product comparisons must use the actual train/impute commands, 10 folds and
seed 77, with validation taken from 5% of the nine training folds. Calculate
metrics only after pooling all ten held-out test folds. Use the same marker
list, split, allele universe, training AF and hard-call conversion for every
method.

Report micro metrics in AF bins `<1%`, `1–5%`, `5–10%`, `10–20%`, `≥20%`,
and a rare-allele table that separates `AF=0`, `0–0.1%`, `0.1–0.5%` and
`0.5–1%`. Historical numbers from other splits or metric definitions are not
evidence of current product performance. Superiority over baseline methods
has not been established by the corrected benchmark. External data used for
parameter selection must not also serve as the final independent test.

## License

MIT — see [LICENSE](LICENSE).

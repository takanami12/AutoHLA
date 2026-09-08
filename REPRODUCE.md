# Reproducing the paper results

Every number in the paper comes from **one command pair**, run once per fold and
per gene group. There is no second code path, no cross-validation-specific
tuning, and no step that reads a test fold.

```bash
autohla train  --vcf  <cell>/fold<F>/train.vcf.gz \
               --labels <labels> --group <G> --marker-list <positions> \
               --phase <off|on> --posthoc-folds 10 --pair on \
               --epochs 100 --s1-epochs 100 --threads 8 \
               --out  <arm>/fold<F>_g<G>/model
autohla impute --model <arm>/fold<F>_g<G>/model \
               --vcf  <cell>/fold<F>/test.vcf.gz \
               --out  <arm>/fold<F>_g<G>/calls.csv
```

`tools/run_cv.py` is a thin loop over `(fold, group)` that issues exactly those
two commands as subprocesses. It contains no modelling logic. Ten-fold
cross-validation uses `--folds 1-10`; an external test uses `--folds 91`, where
fold 91 is "train on the whole cohort, test on the external one". Both go
through the same two commands.

## One command, two behaviours — decided by sample count, not by us

`autohla train` enables its post-hoc layers from `n_train` alone
(`autohla/config.py`, `small = n_train < 4000`):

| | `n_train < 4000` | `n_train >= 4000` |
|---|---|---|
| layers | base + pair + **ridge + blend** + `tau` | base + pair + `tau` |
| blend coefficient `beta` | fitted on **out-of-fold predictions inside the training set** | not created |
| `tau` (homozygote multiplier) | fitted on the 5% validation split of the training set | same |
| cost | 11x (10 inner folds + 1 final model) | 1x + pair fine-tune |

The rule is fixed in the package and applied identically to every cohort. It is
not a per-dataset choice, and nothing in it looks at a test fold. The rule
matches what we measured: on a ~950-sample cohort the ridge layer is worth about
+0.10 F1 in the rarest allele band, while on a ~10,000-sample cohort it *costs*
about 0.07 there — the readout has enough data to learn rare alleles on its own,
and the extra layer only adds noise.

## What is deliberately *not* in this repository

**Sample-level data.** Training genotypes, labels, per-sample calls and truth
tables are identifiable human genetic data and are not redistributable. You need
your own cohort in the layout below. `protocol/*.json` records SHA-256 of every
input file we used, so anyone with authorised access can confirm they hold the
same bytes.

**A second post-hoc path.** Earlier internal work fitted the pair and ridge
layers with a separate script and chose the blend coefficient
*leave-one-fold-out over the other nine test folds*. Those layers themselves
never crossed a fold boundary, but that one scalar did. It is not part of the
published pipeline and its code is not shipped. The package's own
`--posthoc-folds` fits the same coefficient on out-of-fold predictions **inside
the training pool**, which is what the commands above use.

## Data layout

```
data/label/<cohort>.csv                      label table (see README)
data/references/<chip>.position.list         marker whitelist
results/<cell>/data/fold<F>/train.vcf.gz     training genotypes
results/<cell>/data/fold<F>/test.vcf.gz      held-out genotypes
results/<cell>/protocol/fold<F>/*.samples    the split, one sample id per line
```

`scripts/ext_test/prepare.py` builds the folds from raw VCFs, and
`scripts/ext_test/make_ggroup_labels.py` harmonises allele names to G-group
convention when the training and test cohorts were typed by different tools.

## Scoring

```bash
python tools/calls_to_dosage.py --sweep <arm-root> --arm base --name <NAME>
HARDCALL_METHODS=<NAME> python scripts/eval/make_hardcall_pools.py
CV10_COHORT=<COHORT> CV10_CV_DIR=results/<cell> \
CV10_METHODS='AutoHLA=<NAME>_hardcall,...' \
    python scripts/eval/cv10_canonical_metrics.py score
```

Every method — ours and each baseline — passes through the same hard-call rule,
the same allele universe (taken from the cohort label table, never from any
method's predictions) and the same allele-frequency table (counted on each
fold's training samples only). Metrics are reported per allele-frequency bin.
The ten test folds are pooled into one table **before** any metric is computed.

## Checking for leakage yourself

```bash
python scripts/ext_test/audit_leakage.py --cohort <cohort> --folds 1-10 \
    --protocol results/<cell>/protocol \
    --model <arm>/fold01_g1/model --test-vcf <cell>/fold01/test.vcf.gz
```

It exits non-zero if anything fails:

| check | what it rules out |
|---|---|
| A | a fold whose training set overlaps its own test set |
| B | a sample scored in more than one test fold, or in none |
| C | a VCF whose samples disagree with the recorded split |
| F | a call that changes depending on which other samples were imputed with it |
| G | an allele-frequency table estimated from the batch being called |
| H | a model reading markers outside the chip whitelist, or `HLA_*` pseudo-markers smuggled into the training VCF |

The strongest evidence is not a check but an impossibility. `--sandbox` builds a
directory holding **only** `train.vcf.gz`:

```bash
python scripts/ext_test/audit_leakage.py --cohort <cohort> --folds 1 \
    --protocol results/<cell>/protocol --sandbox results/<cell>/data/fold01
# then run `autohla train --vcf <sandbox>/train.vcf.gz ...`
```

If training completes there, it cannot have read a test file — the file is not
on disk. `impute` is a separate process that receives only the test VCF and the
`model/` directory.

## Determinism

Training is not bit-reproducible between runs, even with a fixed seed. On a
single external fold of ~100 samples we measured a spread of 0.031 F1 in the
rarest bin across three runs of identical code on identical data, driven mainly
by `tau` being fitted on a 48-sample validation split. Treat any single number
as carrying at least that much run-to-run variance; the ranking against the
baselines was stable across all runs.

#!/usr/bin/env bash
set -euo pipefail

chip=${1:?usage: materialize_fold.sh CHIP FOLD_NUMBER}
fold=${2:?usage: materialize_fold.sh CHIP FOLD_NUMBER}
root=$(cd "$(dirname "$0")/../.." && pwd)
fold_dir=$(printf '%s/results/cv_han/protocol/fold%02d' "$root" "$fold")
out=$(printf '%s/results/cv_han/data/%s/fold%02d' "$root" "$chip" "$fold")
markers="$root/data/references/ref.$chip.hg18.position.list"
ids="$out/$chip.ids"
mkdir -p "$out"

test -s "$markers" || { echo "missing marker whitelist: $markers" >&2; exit 1; }
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%ID\n' "$root/HAN_dataset/HAN.hg18.vcf.gz" \
  | awk -v list="$markers" '
      BEGIN { FS=OFS="\t" }
      BEGIN {
        while ((getline line < list) > 0) {
          split(line, f, FS)
          want[f[1] FS f[2] FS toupper(f[3]) FS toupper(f[4])] = 1
        }
        close(list)
      }
      { key=$1 FS $2 FS toupper($3) FS toupper($4); if ((key in want) && !(key in seen)) { print $5; seen[key] = 1 } }
    ' - > "$ids"
test "$(wc -l < "$ids")" -gt 0 || { echo "whitelist matched no VCF records" >&2; exit 1; }

for split in train val test; do
  bcftools view -S "$fold_dir/$split.samples" -i "ID=@$ids" -Oz \
    -o "$out/$split.vcf.gz" "$root/HAN_dataset/HAN.hg18.vcf.gz"
  tabix -f -p vcf "$out/$split.vcf.gz"
  cp "$fold_dir/$split.labels.tsv" "$out/$split.labels.tsv"
done

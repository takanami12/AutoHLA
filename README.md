# AutoHLA — VN1K → KHV và HAN reproducibility release

Đây là snapshot sạch của codebase đã sinh ra kết quả external-test tốt nhất
đang có trong workspace cho VN1K → KHV:

- F1 `<1%`: **0.698562**;
- train: 946 mẫu VN1K, test: 99 mẫu KHV;
- 5.032 marker APMDA giao giữa hai VCF;
- seed 77, validation nội bộ 5%, post-hoc 10-fold;
- arm `base`, `--pair off`, `head=full`, `strides=2,2`;
- nhãn đã quy về G-group để khớp quy ước HLA-LA của KHV.

Bảng và hình của kết quả tham chiếu nằm trong
[`reports/vn1k_khv/`](reports/vn1k_khv/). Snapshot mã nguồn là
`f03e684dc7d0e78eb3ee967c440a6dc79ddd81a6`.

Đây là cùng một core codebase cho cả hai cohort. VN1K → KHV là external test;
HAN có hai protocol được giữ lại: HAN 10-fold CV nội bộ (`AutoHLA+pair+ridge`)
và HAN → CHB external. `pair` chỉ là cờ runtime, không phải repo/model riêng.

## Phạm vi public

Repository cố ý không chứa VCF, nhãn cá nhân, sample-level calls, model weights
hoặc output baseline. Người dùng cần có quyền truy cập dữ liệu và đặt chúng vào
đúng layout bên dưới. Hash protocol được ghi tại
[`protocol/vn1k_khv.sha256`](protocol/vn1k_khv.sha256).

## Cài đặt

```bash
python -m pip install '.[benchmark]'
```

Môi trường tham chiếu: Python 3.13.13, torch 2.10.0+cu128, numpy 2.4.2,
pandas 2.3.3, cyvcf2 0.32.1, scipy 1.17.1.

## Layout dữ liệu

Đặt các file private sau trong repository clone:

```text
data/label/ext_VN1K_KHV.ggroup.csv
data/references/ref.APMDA.KHV.position.list
results/cv_vn1k/data/fold91/train.vcf.gz
results/cv_vn1k/data/fold91/test.vcf.gz
results/cv_vn1k/protocol/fold91/train.samples
results/cv_vn1k/protocol/fold91/val.samples
results/cv_vn1k/protocol/fold91/test.samples
```

Để tái lập HAN, bổ sung:

```text
HAN_dataset/HAN.hg18.vcf.gz
HAN_dataset/HAN.HLA.4digit.tsv
data/references/ref.APMDA.hg18.position.list
data/references/ref.APMDA_CHB.hg18.position.list
data/external_test/by_reference/HAN_hg18/CHB.MHC.vcf.gz
data/label/G1K_HLA-LA_Ggroup.resolution.4digits.csv
```

`results/cv_han/data/APMDA/fold01..10/` và
`results/cv_han/data/APMDA_CHB/fold91/` là các VCF đã cắt theo marker whitelist;
`scripts/cv_han/prepare.py` và `scripts/ext_test/prepare.py han_chb` tạo protocol
từ raw data. Các file này không được commit.

Nếu bắt đầu từ raw VN1K/KHV, dùng `scripts/ext_test/prepare.py` để tạo split
fold91 và `scripts/ext_test/make_ggroup_labels.py` để tạo nhãn G-group. Hai công
cụ chỉ nhìn marker nằm trong `ref.APMDA*.position.list`.

## Tái tạo AutoHLA

Chạy trong tmux hoặc job scheduler:

```bash
tmux new -d -s autohla-vn1k-khv \
  'python tools/run_cv.py --cohort vn1k_khv --folds 91 --groups 1-4 \
     --out results/ext_test/vn1k_khv/autohla --par 4 --threads 8 \
     --epochs 100 --s1-epochs 100 --posthoc-folds 10 \
     --pair off --head full'
```

Hoặc gọi trực tiếp hai lệnh `autohla train` / `autohla impute` theo cùng tham số
trong `tools/run_cv.py`. Kết quả gọi phải có 99, 198, 99, 297 dòng cho group
1–4, tổng cộng 693 dòng sample-gene.

## Chấm lại bảng

Gộp calls sang format evaluator:

```bash
python scripts/ext_test/pool.py vn1k_khv --methods AutoHLA
```

Chấm external fold91 bằng cùng evaluator:

```bash
HLA_GGROUP=0 \
CV10_COHORT=VN1K_KHV CV10_FOLDS=91 \
CV10_PRED_DIR=results/ext_test/vn1k_khv/predictions \
CV10_METHODS='AutoHLA=AutoHLA' \
CV10_OUT=results/recomputed/vn1k_khv \
python scripts/eval/cv10_canonical_metrics.py score
```

`HLA_GGROUP=0` là chủ ý: file label đầu vào đã được chuyển G-group trước khi
train. Nếu chấm cùng SNP2HLA/CookHLA/HIBAG/DEEP*HLA, đặt prediction của chúng
vào cùng thư mục rồi dùng:

```bash
CV10_METHODS='SNP2HLA=SNP2HLA,CookHLA=CookHLA,HIBAG=HIBAG,DEEP*HLA=DeepHLA,AutoHLA=AutoHLA' \
CV10_COHORT=VN1K_KHV CV10_FOLDS=91 \
CV10_PRED_DIR=results/ext_test/vn1k_khv/predictions \
CV10_OUT=results/recomputed/vn1k_khv \
python scripts/eval/cv10_canonical_metrics.py score
```

Evaluator dùng chung allele universe, AF theo gene từ train, `TP=min(truth,dosage)`
và chỉ tính sau khi prediction đã được pool. `f1` trong bảng tham chiếu giữ đúng
công thức weighted-by-true-copies của evaluator lịch sử; `sn` và concordance là
micro. Không dùng bảng này để tuyên bố external test độc lập nếu dữ liệu đã được
dùng để chọn tau/G-group.

Vẽ lại line chart từ CSV:

```bash
python -m pip install '.[viz]'
python scripts/plot_vn1k_khv.py
```

## HAN 10-fold CV — kết quả `F1 >=20% = 0.990044`

Tạo manifest dùng chung trước:

```bash
python scripts/cv_han/prepare.py --folds 10 --seed 77
```

Materialize VCF theo đúng manifest (có thể chạy liên tục trong tmux):

```bash
tmux new -d -s autohla-han-materialize \
  "for fold in $(seq 1 10); do bash scripts/cv_han/materialize_fold.sh APMDA \$fold; done"
```

Chạy base trong tmux, sau đó fit pair/ridge và blend:

```bash
tmux new -d -s autohla-han-cv \
  'python tools/run_cv.py --cohort han --folds 1-10 --groups 1-4 \
     --out results/autohla_cv_han --par 4 --threads 8 \
     --epochs 100 --s1-epochs 100 --posthoc-folds 0 --pair off --head full'

POSTHOC_PRED_DIR=results/cv_han/predictions \
python tools/fit_posthoc_cv.py --sweep results/autohla_cv_han --arm base \
  --cohort han --name AUTOHLA_HAN_NS --folds 1-10 --groups 1-4 \
  --threads 8 --pair-epochs 40

BLEND_PRED=results/cv_han/predictions BLEND_COHORT=HAN \
BLEND_PROTOCOL=results/cv_han/protocol \
python scripts/eval/blend_ridge_pair_cv.py \
  --base AUTOHLA_HAN_NS_PAIR --ridge AUTOHLA_HAN_NS_RIDGE_Z \
  --out AUTOHLA_HAN_NS_PAIR_RIDGE --decoder map

HARDCALL_PRED=results/cv_han/predictions \
HARDCALL_METHODS=AUTOHLA_HAN_NS_PAIR_RIDGE \
python scripts/eval/make_hardcall_pools.py

CV10_COHORT=HAN CV10_CV_DIR=results/cv_han \
CV10_PRED_DIR=results/cv_han/predictions \
CV10_METHODS='AutoHLA=AUTOHLA_HAN_NS_PAIR_RIDGE_hardcall,SNP2HLA=HAN_SNP2HLA_hardcall,CookHLA=HAN_COOKHLA_hardcall' \
CV10_OUT=results/recomputed/han_cv \
python scripts/eval/cv10_canonical_metrics.py score
```

Báo cáo tái lập đã đóng gói ở [`reports/han_cv/`](reports/han_cv/) và protocol
đầy đủ ở [`protocol/han_cv.json`](protocol/han_cv.json). Vẽ lại hình bằng:

```bash
python scripts/plot_han_results.py --experiment han_cv
```

## HAN → CHB external

Tạo split fold 91 rồi chạy hai arm để giữ cả so sánh pair:

```bash
python scripts/ext_test/prepare.py han_chb
python scripts/ext_test/make_ggroup_labels.py han_chb

tmux new -d -s autohla-han-chb \
  'python tools/run_cv.py --cohort han_chb --folds 91 --groups 1-4 \
     --out results/ext_test/han_chb/autohla --par 4 --threads 8 \
     --epochs 100 --s1-epochs 100 --posthoc-folds 10 --pair off --head full'

tmux new -d -s autohla-han-chb-pair \
  'python tools/run_cv.py --cohort han_chb --folds 91 --groups 1-4 \
     --out results/ext_test/han_chb/autohla_pair --par 4 --threads 8 \
     --epochs 100 --s1-epochs 100 --posthoc-folds 10 --pair on --head full'

python scripts/ext_test/pool.py han_chb --methods AutoHLA
python scripts/ext_test/pool.py han_chb --methods AutoHLA \
  --autohla-dir results/ext_test/han_chb/autohla_pair/base \
  --autohla-name 'AutoHLA+pair'
```

Chấm cùng evaluator, loại DQA1 đúng như báo cáo external:

```bash
HLA_GGROUP=0 CV10_COHORT=HAN_CHB CV10_FOLDS=91 \
CV10_LOCI=A,B,C,DPB1,DRB1,DQB1 \
CV10_PRED_DIR=results/ext_test/han_chb/predictions \
CV10_METHODS='AutoHLA=AutoHLA,AutoHLA+pair=AutoHLA+pair,SNP2HLA=SNP2HLA,CookHLA=CookHLA,DEEP*HLA=DeepHLA' \
CV10_OUT=results/recomputed/han_chb \
python scripts/eval/cv10_canonical_metrics.py score
```

Bảng tham chiếu và hình nằm ở [`reports/han_chb/`](reports/han_chb/); protocol
đầy đủ ở [`protocol/han_chb.json`](protocol/han_chb.json). Vẽ lại hình bằng:

```bash
python scripts/plot_han_results.py --experiment han_chb
```

## License

MIT — xem [`LICENSE`](LICENSE).

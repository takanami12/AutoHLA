# AutoHLA — VN1K → KHV reproducibility release

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

## License

MIT — xem [`LICENSE`](LICENSE).

# HAN → CHB — external test

Đây là external test: train trên HAN, test trên 103 mẫu CHB 1KGP, fold 91,
2.022 marker chung và 6 gene được chấm. `DQA1` bị loại khỏi bảng vì nhãn external
không hợp lệ; cùng loại bỏ cho mọi method.

| AF bin | AutoHLA | AutoHLA+pair | SNP2HLA | CookHLA | DEEP*HLA |
|---|---:|---:|---:|---:|---:|
| `<1%` | 0.683584 | 0.640789 | 0.752564 | 0.767602 | 0.680634 |
| `1-5%` | 0.934822 | 0.934179 | 0.945932 | 0.935862 | 0.956353 |
| `5-10%` | 0.963290 | 0.963265 | 0.967258 | 0.969661 | 0.968469 |
| `10-20%` | 0.967999 | 0.980427 | 0.955562 | 0.947309 | 0.978076 |
| `>=20%` | 0.980964 | 0.985747 | 0.941321 | 0.920637 | 0.976582 |

Với mục tiêu rare-AF, arm tốt nhất là `AutoHLA` (`--pair off`) ở bin `<1%`.
Arm `AutoHLA+pair` được giữ trong codebase để so sánh trực tiếp; pair không phải
codebase khác mà là cờ runtime.

Bảng AF rare đầy đủ nằm ở
[`summary_by_rare_af_bin.csv`](summary_by_rare_af_bin.csv); bảng raw ở
[`summary_by_af_bin.csv`](summary_by_af_bin.csv), audit ở
[`input_audit.csv`](input_audit.csv), và hình line chart ở
[`Figure_metrics_by_af.svg`](Figure_metrics_by_af.svg).

## Tái lập

Raw HAN/CHB, labels và VCF không nằm trong repository public. Đặt chúng theo
layout trong README, dùng `scripts/ext_test/prepare.py han_chb`, rồi chạy
`tools/run_cv.py --cohort han_chb --folds 91 --groups 1-4` trong tmux. Cấu hình
đầy đủ được ghi ở [`protocol/han_chb.json`](../../protocol/han_chb.json).

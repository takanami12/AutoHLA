# HAN — 10-fold CV

Đây là **CV nội bộ**, không phải external HAN → CHB. Giao thức dùng 10 outer
fold, seed 77, validation 5% của 9-fold train pool, 25.996 marker APMDA hg18 và
đủ 7 gene. Kết quả chính là sau khi gộp cả 10 test fold, hard-call thống nhất,
rồi mới tính metric.

Arm tốt nhất của codebase là `AutoHLA+pair+ridge`; F1 tại bin `>=20%` là
**0.990044** (bảng dưới làm tròn 6 chữ số). Số mẫu dùng cho bảng là 10.488,
theo `input_audit.csv`; con số 8.967 ở một báo cáo cũ là tiêu đề lịch sử và không
phải audit của output này.

| AF bin | AutoHLA+pair+ridge | AutoHLA | AEHLA+pair+ridge | CookHLA | SNP2HLA |
|---|---:|---:|---:|---:|---:|
| `<1%` | 0.688486 | 0.688968 | 0.688748 | 0.721421 | 0.670272 |
| `1-5%` | 0.927550 | 0.918311 | 0.917970 | 0.903282 | 0.898884 |
| `5-10%` | 0.958986 | 0.950017 | 0.948713 | 0.898679 | 0.905926 |
| `10-20%` | 0.968221 | 0.961890 | 0.961458 | 0.917488 | 0.931540 |
| `>=20%` | 0.990044 | 0.987468 | 0.989255 | 0.872140 | 0.912625 |

Bảng AF rare đầy đủ nằm ở
[`summary_by_rare_af_bin.csv`](summary_by_rare_af_bin.csv); bảng raw tổng hợp ở
[`summary_by_af_bin.csv`](summary_by_af_bin.csv), audit ở
[`input_audit.csv`](input_audit.csv), và hình line chart ở
[`Figure_metrics_by_af.svg`](Figure_metrics_by_af.svg).

Lưu ý: `0.990044` chỉ là HAN **CV**. Không được dùng nó làm kết quả external
HAN → CHB; external có 103 mẫu CHB và được báo cáo riêng.

## Tái lập

Raw HAN và các VCF đã bị loại khỏi repository public. Đặt chúng theo layout trong
README, tạo manifest bằng `scripts/cv_han/prepare.py`, sau đó chạy các lệnh trong
[`protocol/han_cv.json`](../../protocol/han_cv.json) trong tmux. Mọi output
sample-level vẫn nằm ngoài git.

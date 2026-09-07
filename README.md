# AutoHLA

Package tối giản để tái hiện giao thức external test **VN1K → KHV** của
AutoHLA: huấn luyện trên VN1K và gọi allele HLA 4-digit cho KHV.

Release này đóng băng đúng source snapshot đã sinh ra kết quả tham chiếu local.
Repository không chứa dữ liệu cá nhân, model weights, results hay test nội bộ.

## Cài đặt

```bash
git clone https://github.com/takanami12/AutoHLA.git
cd AutoHLA
python -m pip install .
```

Reference environment dùng để tái hiện kết quả:

```text
Python 3.13.13
torch 2.10.0+cu128
numpy 2.4.2
pandas 2.3.3
cyvcf2 0.32.1
```

## VN1K → KHV

Chuẩn bị ba file private của protocol:

```text
<VN1K>/train.vcf.gz
<KHV>/test.vcf.gz
<labels>/ext_VN1K_KHV.ggroup.csv
<markers>/ref.APMDA.KHV.position.list
```

Chạy đúng arm base dùng trong kết quả tham chiếu (`--pair off` là bắt buộc):

```bash
export OMP_NUM_THREADS=8

for group in 1 2 3 4; do
  autohla train \
    --vcf <VN1K>/train.vcf.gz \
    --labels <labels>/ext_VN1K_KHV.ggroup.csv \
    --group "$group" \
    --out "model/g${group}" \
    --marker-list <markers>/ref.APMDA.KHV.position.list \
    --phase off \
    --head full \
    --strides 2,2 \
    --epochs 100 \
    --s1-epochs 100 \
    --posthoc-folds 10 \
    --pair off \
    --threads 8

  autohla impute \
    --model "model/g${group}" \
    --vcf <KHV>/test.vcf.gz \
    --out "calls/g${group}.csv" \
    --threads 8
done
```

Protocol cố định: 946 mẫu VN1K để train, validation nội bộ 5% tách từ train,
99 mẫu KHV để test, 5,032 chip markers, seed nội bộ 77 và 10-fold post-hoc.
`val.vcf.gz` của outer split không được truyền vào lệnh train.

Số dòng kết quả mong đợi là 99 cho group 1, 198 cho group 2, 99 cho group 3
và 297 cho group 4 — tổng cộng 693 dòng `(sample_id, gene)`.
Hard allele calls và metrics phải trùng; posterior có thể lệch ở chữ số cuối
giữa các BLAS/CPU khác nhau.

## Định dạng dữ liệu

Label table là CSV/TSV, gồm một cột sample ID và hai cột cho mỗi allele:

```csv
sample_id,A_1,A_2,B_1,B_2
VN001,33:03,11:01,58:01,15:02
```

VCF train và test phải chứa cùng chip markers. Model ghi lại marker list,
allele universe, cấu hình, ridge layer và tham số decoder trong thư mục model.

## License

MIT — xem [LICENSE](LICENSE).

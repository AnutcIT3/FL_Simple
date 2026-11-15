# Báo cáo thí nghiệm Federated Learning

Mô hình **linear**, 7 client, 100 vòng, 5 seed. Dữ liệu sklearn digits, stratified train/validation/test = 60/15/25%.
Số liệu: **mean ± nửa độ rộng khoảng tin cậy 95%** (phân phối t, 5 seed). Mọi so sánh là paired: cùng seed, partition, thứ tự batch của từng client và danh tính kẻ tấn công.

## S – Self-check

| Kiểm tra | Sai số lớn nhất qua các seed | Kết quả |
|---|---:|---|
| FedSGD(IID) == central GD | 3.3e-16 | PASS |
| FedSGD(alpha=0.1) == central GD | 7.0e-16 | PASS |
| aggregate[equal_mean](u,...,u) == u | 8.9e-16 | PASS |
| aggregate[trimmed](u,...,u) == u | 4.4e-16 | PASS |
| aggregate[median](u,...,u) == u | 0.0e+00 | PASS |
| aggregate[krum](u,...,u) == u | 0.0e+00 | PASS |
| aggregate[normclip](u,...,u) == u | 1.3e-13 | PASS |
| aggregate[weighted_mean](u,...,u) == u | 4.4e-16 | PASS |

FedSGD với weighted mean và mọi client tham gia **là** full-batch GD, kể cả khi non-IID. Vì vậy không cần so FedSGD giữa các mức non-IID: chênh lệch luôn bằng 0.

## E1 – IID sạch

| Cấu hình | Test acc (%) |
|---|---:|
| central GD (full-batch) | 93.4 ± 1.0 |
| central mini-batch SGD (exposure-matched) | 96.6 ± 0.6 |
| FedSGD (IID) | 93.4 ± 1.0 |
| FedAvg (IID) | 94.8 ± 0.6 |

## E2 – Client drift: FedAvg theo số epoch cục bộ và mức non-IID

| Cấu hình | Test acc (%) | Client tệ nhất (%) |
|---|---:|---:|
| FedAvg/IID/E=1 | 93.4 ± 1.0 | 92.7 ± 1.2 |
| FedAvg/IID/E=5 | 96.0 ± 0.5 | 95.4 ± 0.8 |
| FedAvg/IID/E=20 | 96.6 ± 0.8 | 96.2 ± 1.0 |
| FedAvg/alpha=0.5/E=1 | 93.2 ± 1.0 | 87.8 ± 5.1 |
| FedAvg/alpha=0.5/E=5 | 95.9 ± 0.6 | 91.8 ± 2.8 |
| FedAvg/alpha=0.5/E=20 | 96.5 ± 0.8 | 93.8 ± 2.8 |
| FedAvg/alpha=0.1/E=1 | 93.0 ± 1.3 | 84.4 ± 6.0 |
| FedAvg/alpha=0.1/E=5 | 95.2 ± 0.9 | 87.9 ± 5.1 |
| FedAvg/alpha=0.1/E=20 | 96.0 ± 0.7 | 91.5 ± 3.4 |

Client tệ nhất = accuracy test có trọng số theo phân bố nhãn của client đó, lấy min qua các client.

## E3 – Chống drift (alpha=0.1, E=20), hyperparameter chọn trên validation

| Cấu hình | Test acc (%) | Client tệ nhất (%) | Cấu hình được chọn (số seed) |
|---|---:|---:|---:|
| FedAvg | 95.9 ± 0.8 | 91.5 ± 3.5 | {"local_lr": 0.05} ×3, {"local_lr": 0.1} ×1, {"local_lr": 0.3} ×1 |
| FedProx | 96.0 ± 0.8 | 91.4 ± 3.4 | {"mu": 0.01} ×3, {"mu": 0.1} ×2 |
| SCAFFOLD | 96.6 ± 0.6 | 93.3 ± 4.4 | {"local_lr": 0.3} ×1, {"local_lr": 0.05} ×3, {"local_lr": 0.1} ×1 |
| FedAvgM | 96.2 ± 0.6 | 91.9 ± 2.4 | {"server_lr": 0.3} ×5 |

Mỗi phương pháp có cùng ngân sách 3 cấu hình (xem TUNING trong code). Test chỉ được đọc cho cấu hình thắng trên validation.

## E3b – Tham gia một phần (alpha=0.1, E=5, hyperparameter mặc định)

| Cấu hình | Test acc (%) | Dao động acc 20 vòng cuối (pp) |
|---|---:|---:|
| FedAvg, tất cả client/vòng | 95.2 ± 0.9 | 0.17 |
| FedAvg, 3/7 client/vòng | 92.7 ± 3.6 | 1.48 |
| SCAFFOLD, tất cả client/vòng | 96.2 ± 0.4 | 0.08 |
| SCAFFOLD, 3/7 client/vòng | 96.2 ± 0.4 | 0.07 |

## E4 – Byzantine robustness

2/7 client độc hại (chọn ngẫu nhiên theo seed, cố định giữa các cách gộp). Server luôn giả định f = 2: trimmed cắt 2 mỗi đầu, Krum dùng f = 2. Hàng `none` vì vậy đo cái giá của phòng thủ khi không ai tấn công.

### Tóm tắt minimax: accuracy sạch / accuracy tệ nhất qua 6 tấn công / ASR backdoor

| Cách gộp | FedSGD IID | FedSGD alpha=0.5 | FedSGD alpha=0.1 | FedAvg IID | FedAvg alpha=0.5 | FedAvg alpha=0.1 |
|---|---:|---:|---:|---:|---:|---:|
| equal_mean | 93 / 10 / 98 | 93 / 10 / 96 | 92 / 10 / 95 | 95 / 10 / 99 | 95 / 10 / 98 | 94 / 11 / 95 |
| trimmed | 93 / 92 / 0 | 90 / 60 / 2 | 57 / 32 / 33 | 94 / 93 / 0 | 92 / 68 / 1 | 68 / 34 / 27 |
| median | 93 / 92 / 0 | 86 / 62 / 3 | 53 / 37 / 34 | 94 / 93 / 0 | 90 / 63 / 1 | 62 / 34 / 22 |
| krum | 93 / 92 / 0 | 73 / 42 / 20 | 47 / 36 / 80 | 93 / 93 / 0 | 72 / 41 / 0 | 44 / 29 / 33 |
| normclip | 93 / 91 / 58 | 93 / 71 / 97 | 89 / 62 / 98 | 95 / 92 / 82 | 95 / 80 / 98 | 92 / 56 / 98 |

ASR nền khi không có backdoor (equal_mean, trung bình các cột): 0.0%. Accuracy tệ nhất có tính cả tấn công backdoor (đo bằng accuracy sạch, không phải ASR).

### FedSGD, IID: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 93.4 ± 1.0 | 92.9 ± 1.0 | 93.2 ± 1.1 | 92.7 ± 0.7 | 93.3 ± 1.1 |
| sign_flip | 10.0 ± 0.2 | 92.0 ± 2.0 | 91.7 ± 2.5 | 92.1 ± 1.7 | 90.7 ± 2.1 |
| gaussian | 93.4 ± 1.0 | 92.9 ± 1.5 | 93.0 ± 1.5 | 92.1 ± 1.7 | 92.8 ± 1.3 |
| alie | 93.3 ± 1.3 | 93.2 ± 1.6 | 93.2 ± 2.0 | 93.5 ± 1.4 | 93.1 ± 1.3 |
| ipm | 92.1 ± 2.0 | 92.3 ± 1.8 | 92.0 ± 2.6 | 91.8 ± 1.5 | 92.0 ± 2.0 |
| label_flip | 89.6 ± 1.9 | 92.1 ± 2.7 | 92.0 ± 2.4 | 92.1 ± 1.7 | 91.6 ± 2.4 |
| backdoor | 92.5 ± 1.5 | 92.7 ± 1.4 | 92.1 ± 1.5 | 92.1 ± 1.7 | 92.5 ± 1.6 |

### FedSGD, alpha=0.5: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 93.1 ± 0.8 | 90.1 ± 4.7 | 86.1 ± 5.5 | 73.2 ± 9.9 | 92.9 ± 0.9 |
| sign_flip | 10.0 ± 0.4 | 67.8 ± 12.4 | 64.8 ± 10.6 | 70.8 ± 16.7 | 71.2 ± 5.7 |
| gaussian | 93.0 ± 1.4 | 91.8 ± 1.9 | 90.2 ± 4.3 | 70.8 ± 16.7 | 92.2 ± 1.5 |
| alie | 92.4 ± 2.1 | 91.9 ± 2.0 | 88.8 ± 5.0 | 92.2 ± 1.8 | 92.4 ± 1.9 |
| ipm | 90.8 ± 2.6 | 60.0 ± 7.5 | 62.2 ± 6.7 | 61.5 ± 13.3 | 88.5 ± 2.7 |
| label_flip | 75.5 ± 5.2 | 70.0 ± 5.0 | 61.8 ± 7.6 | 41.5 ± 42.5 | 74.5 ± 5.4 |
| backdoor | 92.4 ± 1.2 | 90.4 ± 2.4 | 86.6 ± 1.4 | 72.9 ± 11.2 | 91.8 ± 1.2 |

### FedSGD, alpha=0.1: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 92.1 ± 1.2 | 57.2 ± 8.8 | 53.1 ± 8.4 | 46.5 ± 15.2 | 88.9 ± 3.4 |
| sign_flip | 10.1 ± 0.2 | 61.8 ± 10.7 | 48.0 ± 13.3 | 46.4 ± 13.9 | 66.6 ± 6.2 |
| gaussian | 91.1 ± 1.0 | 78.9 ± 9.6 | 62.0 ± 13.6 | 46.4 ± 13.9 | 84.3 ± 9.2 |
| alie | 85.9 ± 5.1 | 81.8 ± 4.6 | 58.8 ± 16.2 | 85.6 ± 4.8 | 84.8 ± 5.1 |
| ipm | 81.7 ± 2.6 | 31.9 ± 11.6 | 36.7 ± 8.5 | 39.7 ± 16.5 | 61.9 ± 11.7 |
| label_flip | 70.4 ± 6.6 | 43.4 ± 12.3 | 37.1 ± 13.3 | 36.0 ± 29.4 | 71.8 ± 6.2 |
| backdoor | 90.9 ± 1.0 | 52.8 ± 4.8 | 48.3 ± 4.2 | 43.7 ± 19.8 | 84.5 ± 7.9 |

### FedAvg, IID: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 94.8 ± 0.6 | 94.4 ± 0.3 | 94.1 ± 0.5 | 93.2 ± 0.6 | 94.6 ± 0.8 |
| sign_flip | 10.0 ± 0.2 | 92.9 ± 1.9 | 93.2 ± 2.5 | 93.5 ± 0.5 | 92.0 ± 2.1 |
| gaussian | 94.7 ± 0.5 | 94.7 ± 0.8 | 94.1 ± 1.0 | 93.5 ± 0.5 | 94.1 ± 0.8 |
| alie | 94.6 ± 0.8 | 94.8 ± 1.0 | 94.7 ± 1.0 | 94.7 ± 0.9 | 94.6 ± 0.9 |
| ipm | 93.4 ± 1.3 | 93.7 ± 1.2 | 92.9 ± 1.8 | 92.8 ± 0.7 | 93.5 ± 1.1 |
| label_flip | 89.8 ± 1.3 | 93.8 ± 1.9 | 93.4 ± 2.1 | 93.5 ± 0.5 | 93.0 ± 2.0 |
| backdoor | 94.2 ± 0.8 | 94.0 ± 0.8 | 93.6 ± 1.1 | 93.5 ± 0.5 | 93.9 ± 0.8 |

### FedAvg, alpha=0.5: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 94.8 ± 0.5 | 92.4 ± 1.9 | 89.7 ± 3.1 | 72.1 ± 8.7 | 94.8 ± 0.4 |
| sign_flip | 9.9 ± 0.4 | 73.5 ± 8.1 | 67.6 ± 10.1 | 76.2 ± 12.2 | 80.4 ± 1.9 |
| gaussian | 94.5 ± 0.6 | 93.4 ± 1.1 | 91.9 ± 2.8 | 76.2 ± 12.2 | 93.8 ± 0.7 |
| alie | 93.7 ± 1.4 | 93.2 ± 1.3 | 90.8 ± 4.1 | 93.6 ± 1.2 | 93.5 ± 1.4 |
| ipm | 92.7 ± 1.3 | 68.0 ± 12.9 | 62.8 ± 13.1 | 50.8 ± 10.3 | 88.1 ± 9.0 |
| label_flip | 75.0 ± 6.3 | 77.6 ± 5.4 | 73.8 ± 7.5 | 41.2 ± 45.8 | 85.1 ± 3.1 |
| backdoor | 94.3 ± 0.7 | 93.0 ± 1.7 | 90.8 ± 2.6 | 74.3 ± 11.7 | 94.3 ± 0.9 |

### FedAvg, alpha=0.1: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 94.0 ± 1.3 | 68.0 ± 11.4 | 62.4 ± 9.7 | 43.7 ± 2.3 | 91.6 ± 2.3 |
| sign_flip | 10.6 ± 1.5 | 62.1 ± 16.6 | 52.4 ± 15.4 | 47.9 ± 18.0 | 69.8 ± 4.4 |
| gaussian | 93.8 ± 0.7 | 81.8 ± 4.5 | 68.4 ± 9.5 | 47.9 ± 18.0 | 86.3 ± 4.9 |
| alie | 85.3 ± 2.3 | 83.3 ± 3.5 | 64.0 ± 15.6 | 85.3 ± 2.7 | 84.8 ± 2.0 |
| ipm | 83.9 ± 1.7 | 33.9 ± 4.8 | 34.1 ± 10.7 | 28.8 ± 13.0 | 56.4 ± 14.4 |
| label_flip | 72.0 ± 11.3 | 48.5 ± 16.9 | 45.1 ± 13.1 | 39.0 ± 17.6 | 74.3 ± 3.1 |
| backdoor | 93.5 ± 1.4 | 62.3 ± 9.0 | 57.4 ± 11.4 | 45.8 ± 20.9 | 90.4 ± 2.2 |

File `paired_e4.csv` có thêm hai đại lượng paired kèm CI 95%: lợi ích so với equal_mean cùng tấn công, và net drop so với hệ thống sạch không phòng thủ (none/equal_mean). Net drop là thứ người dùng thật sự chịu.

## E4b – Khai gian số mẫu khi server dùng weighted mean

| Cấu hình | Test acc (%) |
|---|---:|
| FedSGD/IID, 1 client label-flip khai n x1, weighted_mean | 92.4 ± 2.6 |
| FedSGD/IID, 1 client label-flip khai n x20, weighted_mean | 2.5 ± 1.3 |

## E5 – DP-FedAvg (client-level, mọi client tham gia)

| σ | Test acc (%) | ε (δ = 1e-5) |
|---|---:|---:|
| 0.0 | 94.8 ± 0.6 | inf |
| 0.3 | 94.2 ± 1.2 | 715.5 |
| 1.0 | 88.6 ± 2.6 | 98.0 |
| 3.0 | 64.5 ± 7.5 | 21.6 |

Clip C = 0.5 cố định (chưa dùng adaptive clipping). ε tính bằng RDP composition không có khuếch đại do subsampling, nên là cận trên thô.

## E6 – Rò rỉ thông tin từ update (softmax regression)

- Khôi phục ảnh từ gradient batch=1: sai số lớn nhất 1.1e-16, đoán đúng nhãn 100%.
- Phân bố nhãn của client từ gradient FedSGD vòng 1 (alpha=0.1): sai số L1 lớn nhất 1.1e-14.
- Từ bias của update FedAvg (2 epoch): Spearman trung bình với phân bố nhãn thật 1.00.

## Giới hạn

- Dữ liệu nhỏ, ảnh 8x8, 7 client. K nhỏ khiến DP và partial participation chỉ mang tính minh họa.
- Hyperparameter nền (lr cục bộ 0.1, lr FedSGD 0.5, 2 epoch) kế thừa từ bản đầu, chỉ E3 được tune trên validation.
- Kẻ tấn công ALIE/IPM được giả định biết update của client trung thực; chưa có tấn công thích ứng tối ưu riêng cho từng cách gộp, nên robustness ở đây là cận trên lạc quan.
- Server biết đúng f = f_assumed. Nếu f thật > f_assumed, trimmed và Krum không còn bảo đảm.
- 5 seed: chênh lệch nằm trong CI 95% không nên diễn giải là khác biệt thật.

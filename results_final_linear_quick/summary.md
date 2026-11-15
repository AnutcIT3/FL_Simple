# Báo cáo thí nghiệm Federated Learning

Mô hình **linear**, 7 client, 20 vòng, 2 seed. Dữ liệu sklearn digits, stratified train/validation/test = 60/15/25%.
Số liệu: **mean ± nửa độ rộng khoảng tin cậy 95%** (phân phối t, 2 seed). Mọi so sánh là paired: cùng seed, partition, thứ tự batch của từng client và danh tính kẻ tấn công.

## S – Self-check

| Kiểm tra | Sai số lớn nhất qua các seed | Kết quả |
|---|---:|---|
| FedSGD(IID) == central GD | 1.3e-16 | PASS |
| FedSGD(alpha=0.1) == central GD | 3.3e-16 | PASS |
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
| central GD (full-batch) | 90.8 ± 9.9 |
| central mini-batch SGD (exposure-matched) | 95.1 ± 2.8 |
| FedSGD (IID) | 90.8 ± 9.9 |
| FedAvg (IID) | 92.0 ± 5.6 |

## E2 – Client drift: FedAvg theo số epoch cục bộ và mức non-IID

| Cấu hình | Test acc (%) | Client tệ nhất (%) |
|---|---:|---:|
| FedAvg/IID/E=1 | 90.8 ± 9.9 | 89.1 ± 10.0 |
| FedAvg/IID/E=5 | 93.7 ± 7.1 | 92.8 ± 6.6 |
| FedAvg/IID/E=20 | 95.3 ± 2.8 | 94.6 ± 0.7 |
| FedAvg/alpha=0.5/E=1 | 89.1 ± 19.8 | 72.4 ± 136.3 |
| FedAvg/alpha=0.5/E=5 | 93.8 ± 5.6 | 87.1 ± 29.1 |
| FedAvg/alpha=0.5/E=20 | 95.6 ± 2.8 | 91.5 ± 14.8 |
| FedAvg/alpha=0.1/E=1 | 84.1 ± 49.4 | 59.8 ± 228.3 |
| FedAvg/alpha=0.1/E=5 | 90.7 ± 16.9 | 77.7 ± 127.0 |
| FedAvg/alpha=0.1/E=20 | 92.9 ± 5.6 | 80.4 ± 93.0 |

Client tệ nhất = accuracy test có trọng số theo phân bố nhãn của client đó, lấy min qua các client.

## E3 – Chống drift (alpha=0.1, E=20), hyperparameter chọn trên validation

| Cấu hình | Test acc (%) | Client tệ nhất (%) | Cấu hình được chọn (số seed) |
|---|---:|---:|---:|
| FedAvg | 93.2 ± 12.7 | 79.8 ± 100.1 | {"local_lr": 0.3} ×2 |
| FedProx | 92.8 ± 7.1 | 80.4 ± 93.0 | {"mu": 0.01} ×2 |
| SCAFFOLD | 96.4 ± 5.6 | 93.8 ± 3.3 | {"local_lr": 0.3} ×2 |
| FedAvgM | 95.1 ± 2.8 | 89.6 ± 44.1 | {"server_lr": 1.0} ×2 |

Mỗi phương pháp có cùng ngân sách 3 cấu hình (xem TUNING trong code). Test chỉ được đọc cho cấu hình thắng trên validation.

## E3b – Tham gia một phần (alpha=0.1, E=5, hyperparameter mặc định)

| Cấu hình | Test acc (%) | Dao động acc 20 vòng cuối (pp) |
|---|---:|---:|
| FedAvg, tất cả client/vòng | 90.7 ± 16.9 | 14.78 |
| FedAvg, 3/7 client/vòng | 88.4 ± 2.8 | 20.55 |
| SCAFFOLD, tất cả client/vòng | 93.9 ± 1.4 | 11.91 |
| SCAFFOLD, 3/7 client/vòng | 93.9 ± 4.2 | 23.68 |

## E4 – Byzantine robustness

2/7 client độc hại (chọn ngẫu nhiên theo seed, cố định giữa các cách gộp). Server luôn giả định f = 2: trimmed cắt 2 mỗi đầu, Krum dùng f = 2. Hàng `none` vì vậy đo cái giá của phòng thủ khi không ai tấn công.

### Tóm tắt minimax: accuracy sạch / accuracy tệ nhất qua 6 tấn công / ASR backdoor

| Cách gộp | FedSGD IID | FedSGD alpha=0.5 | FedSGD alpha=0.1 | FedAvg IID | FedAvg alpha=0.5 | FedAvg alpha=0.1 |
|---|---:|---:|---:|---:|---:|---:|
| equal_mean | 91 / 10 / 100 | 88 / 10 / 98 | 73 / 10 / 97 | 92 / 8 / 98 | 92 / 10 / 99 | 91 / 10 / 100 |
| trimmed | 91 / 88 / 0 | 73 / 41 / 15 | 38 / 26 / 73 | 91 / 90 / 0 | 78 / 42 / 3 | 49 / 24 / 63 |
| median | 91 / 88 / 0 | 58 / 41 / 13 | 34 / 28 / 37 | 92 / 90 / 0 | 70 / 36 / 2 | 45 / 18 / 37 |
| krum | 86 / 89 / 0 | 55 / 35 / 50 | 30 / 29 / 50 | 90 / 90 / 0 | 57 / 20 / 0 | 26 / 15 / 19 |
| normclip | 91 / 80 / 57 | 87 / 52 / 100 | 71 / 42 / 99 | 92 / 86 / 67 | 90 / 61 / 98 | 77 / 39 / 100 |

ASR nền khi không có backdoor (equal_mean, trung bình các cột): 0.7%. Accuracy tệ nhất có tính cả tấn công backdoor (đo bằng accuracy sạch, không phải ASR).

### FedSGD, IID: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 90.8 ± 9.9 | 90.6 ± 9.9 | 90.8 ± 7.1 | 85.7 ± 49.4 | 90.8 ± 9.9 |
| sign_flip | 9.9 ± 1.4 | 88.3 ± 4.2 | 88.0 ± 16.9 | 88.7 ± 11.3 | 79.6 ± 31.1 |
| gaussian | 90.4 ± 14.1 | 90.6 ± 15.5 | 90.3 ± 12.7 | 88.7 ± 11.3 | 90.1 ± 12.7 |
| alie | 90.4 ± 14.1 | 90.7 ± 8.5 | 90.8 ± 9.9 | 90.7 ± 8.5 | 90.4 ± 14.1 |
| ipm | 88.4 ± 5.6 | 88.0 ± 11.3 | 87.8 ± 19.8 | 88.7 ± 11.3 | 88.7 ± 8.5 |
| label_flip | 86.8 ± 15.5 | 89.1 ± 22.6 | 88.4 ± 25.4 | 88.7 ± 11.3 | 86.8 ± 9.9 |
| backdoor | 86.0 ± 16.9 | 89.9 ± 18.4 | 89.4 ± 24.0 | 88.7 ± 11.3 | 88.8 ± 9.9 |

### FedSGD, alpha=0.5: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 87.6 ± 25.4 | 72.9 ± 93.2 | 58.3 ± 80.5 | 55.3 ± 22.6 | 86.8 ± 4.2 |
| sign_flip | 10.0 ± 2.8 | 41.1 ± 33.9 | 45.3 ± 101.6 | 48.3 ± 142.6 | 52.1 ± 15.5 |
| gaussian | 85.7 ± 12.7 | 74.3 ± 57.9 | 67.9 ± 114.4 | 48.3 ± 142.6 | 79.9 ± 29.6 |
| alie | 79.7 ± 49.4 | 77.6 ± 0.0 | 69.8 ± 73.4 | 79.1 ± 56.5 | 79.3 ± 39.5 |
| ipm | 69.3 ± 53.6 | 42.6 ± 117.2 | 42.6 ± 94.6 | 45.0 ± 108.7 | 64.1 ± 4.2 |
| label_flip | 64.2 ± 11.3 | 47.2 ± 7.1 | 41.0 ± 35.3 | 35.4 ± 323.3 | 60.6 ± 21.2 |
| backdoor | 80.9 ± 19.8 | 70.9 ± 93.2 | 59.6 ± 2.8 | 54.3 ± 77.6 | 80.1 ± 12.7 |

### FedSGD, alpha=0.1: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 72.9 ± 70.6 | 38.0 ± 5.6 | 34.3 ± 4.2 | 30.0 ± 22.6 | 70.6 ± 128.5 |
| sign_flip | 10.1 ± 1.4 | 49.0 ± 83.3 | 35.2 ± 21.2 | 31.7 ± 57.9 | 41.9 ± 139.8 |
| gaussian | 65.1 ± 59.3 | 58.3 ± 134.1 | 43.0 ± 86.1 | 31.7 ± 57.9 | 60.3 ± 100.2 |
| alie | 71.2 ± 4.2 | 61.2 ± 100.2 | 50.4 ± 0.0 | 75.8 ± 22.6 | 68.0 ± 36.7 |
| ipm | 52.6 ± 80.5 | 26.7 ± 93.2 | 28.1 ± 100.2 | 39.2 ± 4.2 | 46.3 ± 15.5 |
| label_flip | 53.1 ± 31.1 | 36.3 ± 43.8 | 33.0 ± 40.9 | 29.1 ± 33.9 | 51.4 ± 43.8 |
| backdoor | 59.6 ± 90.4 | 26.2 ± 73.4 | 32.0 ± 25.4 | 31.2 ± 7.1 | 64.3 ± 142.6 |

### FedAvg, IID: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 92.0 ± 5.6 | 91.4 ± 7.1 | 91.8 ± 14.1 | 90.4 ± 8.5 | 91.9 ± 7.1 |
| sign_flip | 8.2 ± 19.8 | 90.1 ± 18.4 | 90.3 ± 26.8 | 90.3 ± 9.9 | 86.4 ± 0.0 |
| gaussian | 91.9 ± 1.4 | 91.7 ± 12.7 | 91.7 ± 12.7 | 90.3 ± 9.9 | 91.1 ± 5.6 |
| alie | 91.6 ± 11.3 | 91.6 ± 16.9 | 91.7 ± 15.5 | 91.9 ± 7.1 | 91.3 ± 11.3 |
| ipm | 90.8 ± 18.4 | 91.0 ± 21.2 | 91.0 ± 24.0 | 90.3 ± 9.9 | 90.7 ± 16.9 |
| label_flip | 88.2 ± 14.1 | 90.7 ± 19.8 | 90.8 ± 24.0 | 90.3 ± 9.9 | 89.1 ± 14.1 |
| backdoor | 90.7 ± 8.5 | 91.6 ± 5.6 | 91.4 ± 4.2 | 90.3 ± 9.9 | 90.8 ± 7.1 |

### FedAvg, alpha=0.5: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 91.8 ± 2.8 | 77.9 ± 145.4 | 69.6 ± 118.6 | 56.6 ± 12.7 | 90.2 ± 14.1 |
| sign_flip | 10.0 ± 2.8 | 52.8 ± 57.9 | 49.6 ± 87.5 | 53.6 ± 203.3 | 61.0 ± 55.1 |
| gaussian | 92.3 ± 7.1 | 84.4 ± 56.5 | 79.7 ± 105.9 | 53.6 ± 203.3 | 86.6 ± 32.5 |
| alie | 87.7 ± 7.1 | 85.8 ± 11.3 | 81.7 ± 72.0 | 88.3 ± 15.5 | 86.4 ± 2.8 |
| ipm | 81.0 ± 15.5 | 42.3 ± 55.1 | 35.6 ± 16.9 | 34.8 ± 32.5 | 68.0 ± 79.1 |
| label_flip | 65.6 ± 84.7 | 54.0 ± 5.6 | 52.4 ± 64.9 | 20.0 ± 223.1 | 64.8 ± 35.3 |
| backdoor | 90.8 ± 4.2 | 82.4 ± 81.9 | 75.4 ± 60.7 | 42.7 ± 64.9 | 89.2 ± 15.5 |

### FedAvg, alpha=0.1: test accuracy (%)

| Tấn công | equal_mean | trimmed | median | krum | normclip |
|---|---:|---:|---:|---:|---:|
| none | 90.9 ± 5.6 | 48.9 ± 2.8 | 45.4 ± 24.0 | 25.8 ± 81.9 | 77.1 ± 177.9 |
| sign_flip | 10.0 ± 0.0 | 51.7 ± 74.8 | 40.1 ± 32.5 | 28.8 ± 66.4 | 53.9 ± 35.3 |
| gaussian | 89.3 ± 2.8 | 64.6 ± 69.2 | 52.9 ± 98.8 | 28.8 ± 66.4 | 69.1 ± 107.3 |
| alie | 80.7 ± 11.3 | 71.9 ± 7.1 | 57.7 ± 49.4 | 81.8 ± 8.5 | 79.0 ± 7.1 |
| ipm | 72.3 ± 40.9 | 23.9 ± 55.1 | 18.3 ± 18.4 | 15.0 ± 7.1 | 38.9 ± 19.8 |
| label_flip | 59.0 ± 72.0 | 38.1 ± 60.7 | 40.9 ± 76.2 | 28.8 ± 66.4 | 57.1 ± 73.4 |
| backdoor | 86.6 ± 7.1 | 37.9 ± 7.1 | 45.1 ± 0.0 | 28.8 ± 66.4 | 71.8 ± 160.9 |

File `paired_e4.csv` có thêm hai đại lượng paired kèm CI 95%: lợi ích so với equal_mean cùng tấn công, và net drop so với hệ thống sạch không phòng thủ (none/equal_mean). Net drop là thứ người dùng thật sự chịu.

## E4b – Khai gian số mẫu khi server dùng weighted mean

| Cấu hình | Test acc (%) |
|---|---:|
| FedSGD/IID, 1 client label-flip khai n x1, weighted_mean | 89.6 ± 22.6 |
| FedSGD/IID, 1 client label-flip khai n x20, weighted_mean | 5.4 ± 24.0 |

## E5 – DP-FedAvg (client-level, mọi client tham gia)

| σ | Test acc (%) | ε (δ = 1e-5) |
|---|---:|---:|
| 0.0 | 92.0 ± 5.6 | inf |
| 0.3 | 92.0 ± 8.5 | 182.7 |
| 1.0 | 83.0 ± 12.7 | 31.5 |
| 3.0 | 41.2 ± 114.4 | 8.3 |

Clip C = 0.5 cố định (chưa dùng adaptive clipping). ε tính bằng RDP composition không có khuếch đại do subsampling, nên là cận trên thô.

## E6 – Rò rỉ thông tin từ update (softmax regression)

- Khôi phục ảnh từ gradient batch=1: sai số lớn nhất 1.1e-16, đoán đúng nhãn 100%.
- Phân bố nhãn của client từ gradient FedSGD vòng 1 (alpha=0.1): sai số L1 lớn nhất 9.3e-15.
- Từ bias của update FedAvg (2 epoch): Spearman trung bình với phân bố nhãn thật 1.00.

## Giới hạn

- Dữ liệu nhỏ, ảnh 8x8, 7 client. K nhỏ khiến DP và partial participation chỉ mang tính minh họa.
- Hyperparameter nền (lr cục bộ 0.1, lr FedSGD 0.5, 2 epoch) kế thừa từ bản đầu, chỉ E3 được tune trên validation.
- Kẻ tấn công ALIE/IPM được giả định biết update của client trung thực; chưa có tấn công thích ứng tối ưu riêng cho từng cách gộp, nên robustness ở đây là cận trên lạc quan.
- Server biết đúng f = f_assumed. Nếu f thật > f_assumed, trimmed và Krum không còn bảo đảm.
- 2 seed: chênh lệch nằm trong CI 95% không nên diễn giải là khác biệt thật.

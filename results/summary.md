Ket qua: accuracy cuoi tren tap test (%), trung binh ± do lech chuan qua 5 seed
K = 7 client, 100 vong

## E0
| Cau hinh | Accuracy |
|---|---|
| central GD | 92.8 ± 1.3 |
| FedSGD IID | 92.8 ± 1.3 |

## E1
| Cau hinh | Accuracy |
|---|---|
| hoc tap trung (SGD, moc tran) | 96.0 ± 1.4 |
| FedSGD | 92.8 ± 1.3 |
| FedAvg | 94.2 ± 1.7 |

## E2
| Cau hinh | Accuracy |
|---|---|
| fedsgd (IID) | 92.8 ± 1.3 |
| fedavg (IID) | 94.2 ± 1.7 |
| fedsgd (alpha=1.0) | 92.8 ± 1.3 |
| fedavg (alpha=1.0) | 94.4 ± 1.3 |
| fedsgd (alpha=0.5) | 92.8 ± 1.3 |
| fedavg (alpha=0.5) | 94.8 ± 1.6 |
| fedsgd (alpha=0.1) | 92.8 ± 1.3 |
| fedavg (alpha=0.1) | 93.9 ± 1.4 |

## E3
| Cau hinh | Accuracy |
|---|---|
| f=0, trung bình | 92.8 ± 1.3 |
| f=1, trung bình | 36.6 ± 3.3 |
| f=2, trung bình | 10.0 ± 0.9 |
| f=1, trimmed k=1 | 92.0 ± 1.7 |
| f=2, trimmed k=1 (cắt thiếu) | 10.0 ± 0.9 |
| f=2, trimmed k=2 | 91.0 ± 1.5 |
| f=0, trimmed k=2 (không bị tấn công) | 92.7 ± 1.4 |

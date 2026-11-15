"""Thí nghiệm Federated Learning hợp nhất: hội tụ, client drift, Byzantine robustness, DP và rò rỉ thông tin.

Phụ thuộc: numpy, scipy, scikit-learn, matplotlib (scipy đã đi kèm scikit-learn).

Ví dụ:
    python fl_experiments_final.py --quick                    # smoke test ~1 phút
    python fl_experiments_final.py                            # 5 seed, 100 vòng, softmax regression
    python fl_experiments_final.py --model mlp --workers 4    # mô hình phi tuyến (64-32-10)
    python fl_experiments_final.py --only E1,E4               # chỉ chạy một số thí nghiệm

Thí nghiệm
    S   Self-check: FedSGD weighted == centralized GD trên cả IID lẫn non-IID; các aggregator trả đúng
        vector khi mọi update giống nhau; partition không trùng/thiếu mẫu. Fail -> dừng.
    E1  IID sạch: central GD, central mini-batch SGD (cùng sample exposure với FedAvg), FedSGD, FedAvg.
    E2  Client drift: FedAvg theo số epoch cục bộ E x mức non-IID; báo cả accuracy của client tệ nhất.
    E3  Chống drift (alpha=0.1, E=20): FedAvg, FedProx, SCAFFOLD, FedAvgM, mỗi phương pháp 3 cấu hình,
        chọn trên VALIDATION, báo cáo TEST của cấu hình được chọn. Thêm tham gia một phần (3/7 client).
    E4  Byzantine: {FedSGD, FedAvg} x {IID, alpha=0.5, alpha=0.1} x 7 tấn công x 5 cách gộp.
        Server luôn giả định f=2 (không biết số kẻ tấn công thật). Có backdoor đo bằng attack success rate.
        E4b: kẻ tấn công khai gian số mẫu n khi server dùng weighted mean.
    E5  DP-FedAvg (client-level): clip + nhiễu Gaussian, epsilon tính bằng RDP.
    E6  Rò rỉ: khôi phục ảnh từ gradient batch=1, lộ phân bố nhãn từ update (chỉ với softmax regression).

Thống kê: mọi so sánh dùng chung seed, partition, thứ tự shuffle của từng client và danh tính kẻ tấn công
(paired). Báo cáo mean và khoảng tin cậy 95% theo phân phối t trên các seed.

CẢNH BÁO: benchmark nhỏ (sklearn digits ~1.800 ảnh 8x8, 7 client). Dùng để hiểu cơ chế, không phải bằng
chứng về hiệu quả trên FEMNIST/CIFAR, CNN hay hàng nghìn thiết bị.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_FEATURES, N_CLASSES = 64, 10
TRIGGER = np.array([54, 55, 62, 63])      # 4 pixel góc dưới phải (ảnh 8x8), gần như luôn = 0 trong digits
BACKDOOR_TARGET = 0
ATTACKS = ["none", "sign_flip", "gaussian", "alie", "ipm", "label_flip", "backdoor"]
AGGREGATORS = ["equal_mean", "trimmed", "median", "krum", "normclip"]


# =============================================================================
# Cấu hình
# =============================================================================
@dataclass
class Config:
    rounds: int = 100
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_clients: int = 7
    model: str = "linear"          # linear | mlp
    hidden: int = 32
    local_epochs: int = 2
    batch_size: int = 32
    local_lr: float = 0.1
    fedsgd_lr: float = 0.5
    server_lr: float = 1.0
    f_attack: int = 2              # số kẻ tấn công thật trong E4
    f_assumed: int = 2             # số kẻ tấn công server giả định (trimmed k, Krum f)
    attack_strength: float = 5.0   # sign-flip x5, Gaussian 5 x RMS
    dp_clip: float = 0.5
    dp_sigmas: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0)
    partial_frac: float = 3 / 7
    drift_epochs: tuple[int, ...] = (1, 5, 20)
    only: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6")
    out: str = "results_final"
    workers: int = 2


TUNING = {   # cùng ngân sách: 3 cấu hình mỗi phương pháp, chọn trên validation
    "FedAvg": [{"local_lr": 0.05}, {"local_lr": 0.1}, {"local_lr": 0.3}],
    "FedProx": [{"mu": 0.01}, {"mu": 0.1}, {"mu": 1.0}],
    "SCAFFOLD": [{"local_lr": 0.05}, {"local_lr": 0.1}, {"local_lr": 0.3}],
    "FedAvgM": [{"server_lr": 0.3}, {"server_lr": 0.5}, {"server_lr": 1.0}],
}


def stable_rng(seed: int, *keys: object) -> np.random.Generator:
    """RNG riêng cho từng tình huống: thứ tự vòng lặp không làm đổi kết quả."""
    data = "|".join([str(seed), *map(str, keys)]).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(data).digest()[:8], "little"))


# =============================================================================
# Mô hình: softmax regression hoặc MLP 1 lớp ẩn; tham số là một vector phẳng
# =============================================================================
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Model:
    def __init__(self, kind: str, hidden: int = 32):
        if kind not in ("linear", "mlp"):
            raise ValueError(kind)
        self.kind, self.h = kind, hidden
        self.n_params = ((N_FEATURES + 1) * N_CLASSES if kind == "linear"
                         else (N_FEATURES + 1) * hidden + (hidden + 1) * N_CLASSES)

    def init(self, seed: int) -> np.ndarray:
        if self.kind == "linear":
            return np.zeros(self.n_params)
        r, h = stable_rng(seed, "init", self.h), self.h       # cùng init cho mọi cấu hình của 1 seed
        return np.concatenate([r.normal(0, np.sqrt(2 / N_FEATURES), N_FEATURES * h), np.zeros(h),
                               r.normal(0, np.sqrt(1 / h), h * N_CLASSES), np.zeros(N_CLASSES)])

    def _unpack(self, w):
        h, i = self.h, N_FEATURES * self.h
        return (w[:i].reshape(N_FEATURES, h), w[i:i + h],
                w[i + h:i + h + h * N_CLASSES].reshape(h, N_CLASSES), w[i + h + h * N_CLASSES:])

    def probs(self, w, X):
        if self.kind == "linear":
            return softmax(X @ w[:N_FEATURES * N_CLASSES].reshape(N_FEATURES, N_CLASSES) + w[-N_CLASSES:])
        W1, b1, W2, b2 = self._unpack(w)
        return softmax(np.maximum(X @ W1 + b1, 0) @ W2 + b2)

    def grad(self, w, X, y):
        """Gradient cross-entropy trung bình trên batch, không regularization."""
        if len(y) == 0:
            raise ValueError("Batch rỗng.")
        if self.kind == "linear":
            E = self.probs(w, X)
            E[np.arange(len(y)), y] -= 1.0
            return np.concatenate([(X.T @ E / len(y)).ravel(), E.mean(axis=0)])
        W1, b1, W2, b2 = self._unpack(w)
        Z = X @ W1 + b1
        H = np.maximum(Z, 0)
        P = softmax(H @ W2 + b2)
        P[np.arange(len(y)), y] -= 1.0
        P /= len(y)
        dH = P @ W2.T
        dH[Z <= 0] = 0
        return np.concatenate([(X.T @ dH).ravel(), dH.sum(0), (H.T @ P).ravel(), P.sum(0)])


# =============================================================================
# Dữ liệu
# =============================================================================
@dataclass
class Data:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


def load_data(seed: int) -> Data:
    """Stratified: 25% test; phần còn lại chia 80% train / 20% validation."""
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.2, random_state=seed, stratify=ya)
    return Data(Xtr, ytr, Xva, yva, Xte, yte)


def partition_iid(n: int, k: int, rng) -> list[np.ndarray]:
    return [a.astype(np.int64) for a in np.array_split(rng.permutation(n), k)]


def partition_dirichlet(y, k, alpha, rng, min_size=10, max_tries=5000) -> list[np.ndarray]:
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(k)]
        for c in np.unique(y):
            ids = np.where(y == c)[0].copy()
            rng.shuffle(ids)
            counts = rng.multinomial(len(ids), rng.dirichlet(np.full(k, alpha)))
            for j, chunk in enumerate(np.split(ids, np.cumsum(counts)[:-1])):
                parts[j].extend(map(int, chunk))
        if min(map(len, parts)) >= min_size:
            return [np.asarray(p, dtype=np.int64) for p in parts]
    raise RuntimeError(f"Không chia được Dirichlet alpha={alpha}.")


def make_partitions(d: Data, cfg: Config, seed: int) -> dict[str, list[np.ndarray]]:
    parts = {"IID": partition_iid(len(d.ytr), cfg.n_clients, stable_rng(seed, "partition", "IID"))}
    for a in (0.5, 0.1):
        parts[f"alpha={a}"] = partition_dirichlet(d.ytr, cfg.n_clients, a, stable_rng(seed, "partition", a))
    for name, p in parts.items():
        ids = np.sort(np.concatenate(p))
        if len(ids) != len(d.ytr) or not np.array_equal(ids, np.arange(len(d.ytr))):
            raise AssertionError(f"Partition {name} trùng hoặc thiếu mẫu.")
    return parts


def add_trigger(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, TRIGGER] = 1.0
    return X


# =============================================================================
# Đánh giá
# =============================================================================
def class_accuracy(model, w, X, y) -> np.ndarray:
    pred = model.probs(w, X).argmax(1)
    return np.array([np.mean(pred[y == c] == c) for c in range(N_CLASSES)])


def final_metrics(model, w, d: Data, client_dists: np.ndarray) -> dict[str, float]:
    P = model.probs(w, d.Xte)
    acc = float(np.mean(P.argmax(1) == d.yte))
    loss = float(-np.log(np.clip(P[np.arange(len(d.yte)), d.yte], 1e-15, 1)).mean())
    ca = class_accuracy(model, w, d.Xte, d.yte)
    # accuracy test có trọng số theo phân bố nhãn của từng client: client "tệ nhất" thấy model thế nào
    worst = float(np.min(client_dists @ ca))
    mask = d.yte != BACKDOOR_TARGET
    asr = float(np.mean(model.probs(w, add_trigger(d.Xte[mask])).argmax(1) == BACKDOOR_TARGET))
    return {"test_acc": acc, "test_loss": loss, "worst_client_acc": worst, "backdoor_asr": asr}


def round_metrics(model, w, d: Data) -> tuple[float, float, float]:
    te = float(np.mean(model.probs(w, d.Xte).argmax(1) == d.yte))
    va = float(np.mean(model.probs(w, d.Xva).argmax(1) == d.yva))
    P = model.probs(w, d.Xtr)
    tl = float(-np.log(np.clip(P[np.arange(len(d.ytr)), d.ytr], 1e-15, 1)).mean())
    return te, va, tl


# =============================================================================
# Huấn luyện cục bộ
# =============================================================================
def local_sgd(model, w, X, y, rng, epochs, lr, batch, mu=0.0, corr=None):
    wk, steps = w.copy(), 0
    for _ in range(epochs):
        for b in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / batch))):
            g = model.grad(wk, X[b], y[b])
            if mu:
                g = g + mu * (wk - w)          # FedProx: proximal term kéo về model toàn cục
            if corr is not None:
                g = g + corr                   # SCAFFOLD: c - c_k bù độ lệch của client
            wk -= lr * g
            steps += 1
    return wk, steps


# =============================================================================
# Gộp ở server
# =============================================================================
def aggregate(U: np.ndarray, sizes: np.ndarray, mode: str, f_assumed: int) -> np.ndarray:
    if mode == "weighted_mean":
        return (sizes / sizes.sum()) @ U
    if mode == "equal_mean":
        return U.mean(0)
    if mode == "trimmed":                      # Yin et al. 2018, cắt k = f_assumed mỗi đầu
        if 2 * f_assumed >= len(U):
            raise ValueError("2k phải < số client.")
        return np.sort(U, axis=0)[f_assumed:len(U) - f_assumed].mean(0)
    if mode == "median":
        return np.median(U, axis=0)
    if mode == "krum":                         # Blanchard et al. 2017
        m = len(U) - f_assumed - 2
        if m < 1:
            raise ValueError("Krum cần n >= f + 3.")
        D = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
        return U[np.argmin(np.sort(D, axis=1)[:, 1:m + 1].sum(1))]
    if mode == "normclip":                     # clip về norm trung vị rồi lấy trung bình
        norms = np.linalg.norm(U, axis=1)
        return (U * np.minimum(1.0, np.median(norms) / (norms + 1e-12))[:, None]).mean(0)
    raise ValueError(mode)


def alie_z(n: int, f: int) -> float:
    """Baruch et al. 2019."""
    s = n // 2 + 1 - f
    return NormalDist().inv_cdf((n - s) / n)


def model_poison(U: np.ndarray, attacked: list[int], attack: str, strength: float, rng) -> np.ndarray:
    """Tấn công trên update (model poisoning). Kẻ tấn công biết update của client trung thực (omniscient)."""
    if not attacked or attack in ("none", "label_flip", "backdoor"):
        return U
    U = U.copy()
    honest = np.delete(U, attacked, axis=0)
    for k in attacked:
        if attack == "sign_flip":
            U[k] = -strength * U[k]
        elif attack == "gaussian":
            U[k] = U[k] + rng.normal(0, strength * np.linalg.norm(U[k]) / np.sqrt(U.shape[1]), U.shape[1])
        elif attack == "alie":
            U[k] = honest.mean(0) - alie_z(len(U), len(attacked)) * honest.std(0)
        elif attack == "ipm":                  # Xie et al. 2019, epsilon = 0.5
            U[k] = -0.5 * honest.mean(0)
        else:
            raise ValueError(attack)
    return U


def client_data(X, y, attack: str):
    """Tấn công trên dữ liệu (data poisoning) của client độc hại."""
    if attack == "label_flip":
        return X, 9 - y
    if attack == "backdoor":                   # thêm bản sao có trigger, gán nhãn đích
        m = y != BACKDOOR_TARGET
        return np.vstack([X, add_trigger(X[m])]), np.concatenate([y, np.full(m.sum(), BACKDOOR_TARGET)])
    return X, y


# =============================================================================
# Một lần chạy FL
# =============================================================================
@dataclass
class Run:
    algo: str = "FedAvg"                       # FedSGD | FedAvg | FedProx | SCAFFOLD | FedAvgM
    partition: str = "IID"
    agg: str = "weighted_mean"
    attack: str = "none"
    f: int = 0
    local_epochs: int | None = None
    local_lr: float | None = None
    server_lr: float | None = None
    mu: float = 0.1
    beta: float = 0.9
    frac: float = 1.0
    dp_sigma: float | None = None
    n_lie: float = 1.0                         # hệ số khai gian số mẫu của kẻ tấn công


def run_fl(model: Model, d: Data, parts, cfg: Config, seed: int, r: Run):
    K = len(parts)
    E = r.local_epochs or cfg.local_epochs
    lr = r.local_lr or cfg.local_lr
    eta = r.server_lr or (cfg.fedsgd_lr if r.algo == "FedSGD" else cfg.server_lr)
    attacked = [int(k) for k in stable_rng(seed, "attackers", r.partition).permutation(K)[:r.f]]
    attack = r.attack if r.f > 0 else "none"
    clients = [client_data(d.Xtr[p], d.ytr[p], attack if k in attacked else "none") for k, p in enumerate(parts)]
    sizes = np.array([len(p) for p in parts], float)
    sizes[attacked] *= r.n_lie
    dists = np.stack([np.bincount(d.ytr[p], minlength=N_CLASSES) / len(p) for p in parts])
    # mỗi client một luồng rng riêng -> client trung thực có cùng thứ tự batch ở mọi cấu hình (paired)
    crng = [stable_rng(seed, "local", r.partition, r.algo, k) for k in range(K)]
    arng = stable_rng(seed, "attack_noise", r.partition, r.algo, attack)
    prng = stable_rng(seed, "participation", r.partition, r.frac)
    nrng = stable_rng(seed, "dp_noise", r.partition, r.dp_sigma)
    m_sel = max(1, int(round(r.frac * K)))

    w = model.init(seed)
    c, mom = np.zeros_like(w), np.zeros_like(w)
    ck = [np.zeros_like(w) for _ in range(K)]
    trace, diverged = [], False
    for _ in range(cfg.rounds):
        S = np.arange(K) if m_sel == K else np.sort(prng.choice(K, m_sel, replace=False))
        U, dcs = [], []
        for k in S:
            X, y = clients[k]
            if r.algo == "FedSGD":
                U.append(model.grad(w, X, y))
                continue
            if r.algo == "SCAFFOLD":
                wk, steps = local_sgd(model, w, X, y, crng[k], E, lr, cfg.batch_size, corr=c - ck[k])
                new_ck = ck[k] - c + (w - wk) / (steps * lr)
                dcs.append(new_ck - ck[k])
                ck[k] = new_ck
            else:
                wk, _ = local_sgd(model, w, X, y, crng[k], E, lr, cfg.batch_size,
                                  mu=r.mu if r.algo == "FedProx" else 0.0)
            U.append(w - wk)
        U = np.stack(U)
        att_in_S = [i for i, k in enumerate(S) if k in attacked]
        U = model_poison(U, att_in_S, attack, cfg.attack_strength, arng)
        if r.dp_sigma is not None:             # DP-FedAvg: clip từng client, nhiễu trên tổng
            U = U * np.minimum(1.0, cfg.dp_clip / (np.linalg.norm(U, axis=1) + 1e-12))[:, None]
            step = (U.sum(0) + (nrng.normal(0, r.dp_sigma * cfg.dp_clip, U.shape[1]) if r.dp_sigma else 0)) / len(S)
        else:
            step = aggregate(U, sizes[S], r.agg, cfg.f_assumed)
        if r.algo == "FedAvgM":
            mom = r.beta * mom + step
            step = mom
        w = w - eta * step
        if r.algo == "SCAFFOLD":
            c = c + (sizes[S] / sizes.sum()) @ np.stack(dcs)
        if not np.all(np.isfinite(w)):
            diverged = True
            break
        trace.append(round_metrics(model, w, d))
    if diverged:                               # điền phần còn lại bằng mức đoán bừa
        chance = float(np.mean(d.yte == np.bincount(d.ytr).argmax()))
        trace += [(chance, chance, float("inf"))] * (cfg.rounds - len(trace))
        fm = {"test_acc": chance, "test_loss": float("inf"), "worst_client_acc": 0.0, "backdoor_asr": float("nan")}
    else:
        fm = final_metrics(model, w, d, dists)
    fm.update({"val_acc_tail": float(np.mean([t[1] for t in trace[-5:]])),
               "osc_last20": float(np.std([t[0] for t in trace[-20:]])),
               "diverged": diverged, "attacked": json.dumps(attacked)})
    return w, trace, fm


def run_central(model, d: Data, cfg: Config, seed: int, minibatch: bool):
    w = model.init(seed)
    rng = stable_rng(seed, "central_minibatch")
    trace = []
    for _ in range(cfg.rounds):
        if minibatch:                          # cùng sample exposure với FedAvg: local_epochs epoch / vòng
            w, _ = local_sgd(model, w, d.Xtr, d.ytr, rng, cfg.local_epochs, cfg.local_lr, cfg.batch_size)
        else:
            w = w - cfg.fedsgd_lr * model.grad(w, d.Xtr, d.ytr)
        trace.append(round_metrics(model, w, d))
    full = np.full((1, N_CLASSES), 1 / N_CLASSES)
    fm = final_metrics(model, w, d, full)
    fm.update({"val_acc_tail": float(np.mean([t[1] for t in trace[-5:]])),
               "osc_last20": float(np.std([t[0] for t in trace[-20:]])), "diverged": False, "attacked": "[]"})
    return w, trace, fm


# =============================================================================
# DP accounting và rò rỉ
# =============================================================================
def eps_rdp(sigma: float, T: int, delta: float = 1e-5) -> float:
    """Gaussian mechanism lặp T lần, mọi client tham gia (không có khuếch đại do subsampling)."""
    if sigma <= 0:
        return float("inf")
    a = np.linspace(1.01, 1024, 100000)
    return float(np.min(T * a / (2 * sigma ** 2) + np.log(1 / delta) / (a - 1)))


def leakage(model: Model, d: Data, parts_noniid, cfg: Config, seed: int) -> dict[str, Any]:
    r = stable_rng(seed, "leak")
    w = r.normal(0, 0.1, model.n_params)
    idx = r.choice(len(d.ytr), 6, replace=False)
    recs, errs, labs = [], [], []
    for i in idx:
        g = model.grad(w, d.Xtr[i:i + 1], d.ytr[i:i + 1])
        dW, db = g[:640].reshape(64, 10), g[640:]
        c = int(np.argmin(db))                 # chỉ lớp đúng có db < 0
        x = dW[:, c] / db[c]                   # dW = x (p - e_y)^T  =>  khôi phục x chính xác
        recs.append((d.Xtr[i], x, int(d.ytr[i])))
        errs.append(float(np.abs(x - d.Xtr[i]).max()))
        labs.append(c == d.ytr[i])
    l1, rho = [], []
    for k, p in enumerate(parts_noniid):
        true = np.bincount(d.ytr[p], minlength=10) / len(p)
        g0 = model.grad(np.zeros(model.n_params), d.Xtr[p], d.ytr[p])
        l1.append(float(np.abs((0.1 - g0[640:]) - true).sum()))   # vòng 1, w=0: db = 0.1 - tỷ lệ nhãn
        wk, _ = local_sgd(model, np.zeros(model.n_params), d.Xtr[p], d.ytr[p], stable_rng(seed, "leak", k),
                          cfg.local_epochs, cfg.local_lr, cfg.batch_size)
        rho.append(float(stats.spearmanr(wk[640:], true)[0]))
    return {"img_err": max(errs), "label_acc": float(np.mean(labs)), "hist_l1": max(l1),
            "rho_fedavg": float(np.mean(rho)), "imgs": recs}


# =============================================================================
# Self-check
# =============================================================================
def self_checks(model, d: Data, parts, cfg: Config, seed: int) -> list[dict[str, Any]]:
    out = []
    w_c, _, _ = run_central(model, d, cfg, seed, minibatch=False)
    for p in ("IID", "alpha=0.1"):
        w_f, _, _ = run_fl(model, d, parts[p], cfg, seed, Run(algo="FedSGD", partition=p))
        err = float(np.max(np.abs(w_c - w_f)))
        out.append({"seed": seed, "check": f"FedSGD({p}) == central GD", "value": err, "passed": err < 1e-9})
    v = stable_rng(seed, "check").normal(size=model.n_params)
    U = np.tile(v, (cfg.n_clients, 1))
    for a in AGGREGATORS + ["weighted_mean"]:
        err = float(np.max(np.abs(aggregate(U, np.arange(1, cfg.n_clients + 1, dtype=float), a, cfg.f_assumed) - v)))
        out.append({"seed": seed, "check": f"aggregate[{a}](u,...,u) == u", "value": err, "passed": err < 1e-12})
    for c in out:
        if not c["passed"]:
            raise AssertionError(f"Self-check fail: {c}")
    return out


# =============================================================================
# Toàn bộ thí nghiệm của một seed (chạy song song theo seed)
# =============================================================================
def run_seed(args: tuple[Config, int]) -> dict[str, Any]:
    cfg, seed = args
    t0 = time.time()
    model = Model(cfg.model, cfg.hidden)
    d = load_data(seed)
    parts = make_partitions(d, cfg, seed)
    final, curves, part_rows, extra = [], [], [], {}

    def save(exp, setting, trace, fm, **kw):
        final.append({"experiment": exp, "setting": setting, "seed": seed, **fm, **kw})
        for t, (te, va, tl) in enumerate(trace, 1):
            curves.append({"experiment": exp, "setting": setting, "seed": seed, "round": t,
                           "test_acc": te, "val_acc": va, "train_loss": tl})

    for name, p in parts.items():
        for k, ids in enumerate(p):
            cnt = np.bincount(d.ytr[ids], minlength=N_CLASSES)
            part_rows.append({"seed": seed, "partition": name, "client_id": k, "n_samples": len(ids),
                              "n_classes": int((cnt > 0).sum()), "class_counts": json.dumps(cnt.tolist())})
    checks = self_checks(model, d, parts, cfg, seed)

    if "E1" in cfg.only:
        for name, mb in (("central GD (full-batch)", False), ("central mini-batch SGD (exposure-matched)", True)):
            _, tr, fm = run_central(model, d, cfg, seed, mb)
            save("E1", name, tr, fm)
        for algo in ("FedSGD", "FedAvg"):
            _, tr, fm = run_fl(model, d, parts["IID"], cfg, seed, Run(algo=algo, partition="IID"))
            save("E1", f"{algo} (IID)", tr, fm)

    if "E2" in cfg.only:
        for p in ("IID", "alpha=0.5", "alpha=0.1"):
            for E in cfg.drift_epochs:
                _, tr, fm = run_fl(model, d, parts[p], cfg, seed, Run(algo="FedAvg", partition=p, local_epochs=E))
                save("E2", f"FedAvg/{p}/E={E}", tr, fm, partition=p, local_epochs=E)

    if "E3" in cfg.only:
        for algo, cands in TUNING.items():
            best = None
            for hp in cands:
                _, tr, fm = run_fl(model, d, parts["alpha=0.1"], cfg, seed,
                                   Run(algo=algo, partition="alpha=0.1", local_epochs=20, **hp))
                save("E3-tuning", f"{algo} {json.dumps(hp)}", tr, fm)
                if best is None or fm["val_acc_tail"] > best[2]["val_acc_tail"]:
                    best = (hp, tr, fm)
            save("E3", algo, best[1], best[2], chosen=json.dumps(best[0]))
        for algo in ("FedAvg", "SCAFFOLD"):
            for frac in (1.0, cfg.partial_frac):
                lab = f"{algo}, {'tất cả' if frac == 1 else f'{round(frac * cfg.n_clients)}/{cfg.n_clients}'} client/vòng"
                _, tr, fm = run_fl(model, d, parts["alpha=0.1"], cfg, seed,
                                   Run(algo=algo, partition="alpha=0.1", local_epochs=5, frac=frac))
                save("E3-partial", lab, tr, fm)

    if "E4" in cfg.only:
        for algo in ("FedSGD", "FedAvg"):
            for p in ("IID", "alpha=0.5", "alpha=0.1"):
                for atk in ATTACKS:
                    for agg in AGGREGATORS:
                        f = 0 if atk == "none" else cfg.f_attack
                        _, tr, fm = run_fl(model, d, parts[p], cfg, seed,
                                           Run(algo=algo, partition=p, agg=agg, attack=atk, f=f))
                        save("E4", f"{algo}/{p}/{atk}/{agg}", tr, fm, algo=algo, partition=p, attack=atk, agg=agg)
        for lie in (1.0, 20.0):
            _, tr, fm = run_fl(model, d, parts["IID"], cfg, seed,
                               Run(algo="FedSGD", partition="IID", agg="weighted_mean", attack="label_flip", f=1, n_lie=lie))
            save("E4b", f"FedSGD/IID, 1 client label-flip khai n x{lie:g}, weighted_mean", tr, fm)

    if "E5" in cfg.only:
        for s in cfg.dp_sigmas:
            _, tr, fm = run_fl(model, d, parts["IID"], cfg, seed, Run(algo="FedAvg", partition="IID", dp_sigma=s))
            save("E5", f"sigma={s}", tr, fm, sigma=s)

    if "E6" in cfg.only and cfg.model == "linear":
        extra["leak"] = leakage(model, d, parts["alpha=0.1"], cfg, seed)

    print(f"[seed {seed}] xong trong {time.time() - t0:.0f}s", flush=True)
    return {"final": final, "curves": curves, "parts": part_rows, "checks": checks, "extra": extra}


# =============================================================================
# Thống kê và báo cáo
# =============================================================================
def ci(v) -> tuple[float, float]:
    """Mean và nửa độ rộng khoảng tin cậy 95% (t-distribution)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def fmt(v, scale=100, digits=1) -> str:
    m, h = ci(np.asarray(v) * scale)
    return f"{m:.{digits}f} ± {h:.{digits}f}"


def by(final, exp, setting, key="test_acc") -> list[float]:
    """Gom theo (experiment, setting) -> không đếm trùng giữa các thí nghiệm."""
    rows = sorted((r for r in final if r["experiment"] == exp and r["setting"] == setting), key=lambda r: r["seed"])
    return [r[key] for r in rows]


def paired_tables(final) -> list[dict[str, Any]]:
    """E4: (1) lợi ích phòng thủ so với equal_mean cùng tấn công; (2) net drop so với hệ thống sạch equal_mean."""
    out = []
    for algo in ("FedSGD", "FedAvg"):
        for p in ("IID", "alpha=0.5", "alpha=0.1"):
            clean = np.array(by(final, "E4", f"{algo}/{p}/none/equal_mean"))
            for atk in ATTACKS:
                base = np.array(by(final, "E4", f"{algo}/{p}/{atk}/equal_mean"))
                for agg in AGGREGATORS:
                    acc = np.array(by(final, "E4", f"{algo}/{p}/{atk}/{agg}"))
                    if len(acc) == 0:
                        continue
                    g, gh = ci(100 * (acc - base))
                    nd, nh = ci(100 * (clean - acc))
                    out.append({"algo": algo, "partition": p, "attack": atk, "aggregator": agg,
                                "acc_mean": 100 * acc.mean(), "gain_vs_equal_mean_pp": g, "gain_ci95": gh,
                                "net_drop_vs_clean_mean_pp": nd, "net_drop_ci95": nh})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def report(cfg: Config, final, checks, paired, leaks, out: Path) -> None:
    n = len(cfg.seeds)
    L = ["# Báo cáo thí nghiệm Federated Learning", "",
         f"Mô hình **{cfg.model}**{f' ({cfg.hidden} unit ẩn)' if cfg.model == 'mlp' else ''}, {cfg.n_clients} client, "
         f"{cfg.rounds} vòng, {n} seed. Dữ liệu sklearn digits, stratified train/validation/test = 60/15/25%.",
         f"Số liệu: **mean ± nửa độ rộng khoảng tin cậy 95%** (phân phối t, {n} seed). Mọi so sánh là paired: "
         "cùng seed, partition, thứ tự batch của từng client và danh tính kẻ tấn công.", ""]

    L += ["## S – Self-check", "", "| Kiểm tra | Sai số lớn nhất qua các seed | Kết quả |", "|---|---:|---|"]
    for name in dict.fromkeys(c["check"] for c in checks):
        cs = [c for c in checks if c["check"] == name]
        L.append(f"| {name} | {max(c['value'] for c in cs):.1e} | {'PASS' if all(c['passed'] for c in cs) else 'FAIL'} |")
    L += ["", "FedSGD với weighted mean và mọi client tham gia **là** full-batch GD, kể cả khi non-IID. "
          "Vì vậy không cần so FedSGD giữa các mức non-IID: chênh lệch luôn bằng 0.", ""]

    def simple(exp, title, cols=(("test_acc", "Test acc (%)"),), note=""):
        settings = list(dict.fromkeys(r["setting"] for r in final if r["experiment"] == exp))
        if not settings:
            return
        L.extend([f"## {title}", "", "| Cấu hình | " + " | ".join(c[1] for c in cols) + " |",
                  "|---" + "|---:" * len(cols) + "|"])
        for s in settings:
            cells = []
            for key, _ in cols:
                v = by(final, exp, s, key)
                if key == "chosen":
                    cells.append(", ".join(f"{h} ×{v.count(h)}" for h in dict.fromkeys(v)))
                elif key == "osc_last20":
                    cells.append(f"{100 * np.mean(v):.2f}")
                elif key == "train_loss":
                    cells.append(f"{np.mean(v):.3f}")
                else:
                    cells.append(fmt(v))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        if note:
            L.extend(["", note])
        L.append("")

    simple("E1", "E1 – IID sạch")
    simple("E2", "E2 – Client drift: FedAvg theo số epoch cục bộ và mức non-IID",
           (("test_acc", "Test acc (%)"), ("worst_client_acc", "Client tệ nhất (%)")),
           "Client tệ nhất = accuracy test có trọng số theo phân bố nhãn của client đó, lấy min qua các client.")
    simple("E3", "E3 – Chống drift (alpha=0.1, E=20), hyperparameter chọn trên validation",
           (("test_acc", "Test acc (%)"), ("worst_client_acc", "Client tệ nhất (%)"), ("chosen", "Cấu hình được chọn (số seed)")),
           "Mỗi phương pháp có cùng ngân sách 3 cấu hình (xem TUNING trong code). Test chỉ được đọc cho cấu hình thắng trên validation.")
    simple("E3-partial", "E3b – Tham gia một phần (alpha=0.1, E=5, hyperparameter mặc định)",
           (("test_acc", "Test acc (%)"), ("osc_last20", "Dao động acc 20 vòng cuối (pp)")))

    # E4
    if paired:
        L += ["## E4 – Byzantine robustness", "",
              f"{cfg.f_attack}/{cfg.n_clients} client độc hại (chọn ngẫu nhiên theo seed, cố định giữa các cách gộp). "
              f"Server luôn giả định f = {cfg.f_assumed}: trimmed cắt {cfg.f_assumed} mỗi đầu, Krum dùng f = {cfg.f_assumed}. "
              "Hàng `none` vì vậy đo cái giá của phòng thủ khi không ai tấn công.", "",
              "### Tóm tắt minimax: accuracy sạch / accuracy tệ nhất qua 6 tấn công / ASR backdoor", ""]
        combos = [(a, p) for a in ("FedSGD", "FedAvg") for p in ("IID", "alpha=0.5", "alpha=0.1")]
        L += ["| Cách gộp | " + " | ".join(f"{a} {p}" for a, p in combos) + " |", "|---" + "|---:" * len(combos) + "|"]
        for agg in AGGREGATORS:
            cells = []
            for a, p in combos:
                accs = {r["attack"]: r["acc_mean"] for r in paired if r["algo"] == a and r["partition"] == p and r["aggregator"] == agg}
                asr = 100 * np.mean(by(final, "E4", f"{a}/{p}/backdoor/{agg}", "backdoor_asr"))
                worst = min(v for k, v in accs.items() if k != "none")
                cells.append(f"{accs['none']:.0f} / {worst:.0f} / {asr:.0f}")
            L.append(f"| {agg} | " + " | ".join(cells) + " |")
        clean_asr = np.mean([100 * np.mean(by(final, "E4", f"{a}/{p}/none/equal_mean", "backdoor_asr")) for a, p in combos])
        L += ["", f"ASR nền khi không có backdoor (equal_mean, trung bình các cột): {clean_asr:.1f}%. "
              "Accuracy tệ nhất có tính cả tấn công backdoor (đo bằng accuracy sạch, không phải ASR).", ""]
        for a, p in combos:
            L += [f"### {a}, {p}: test accuracy (%)", "", "| Tấn công | " + " | ".join(AGGREGATORS) + " |",
                  "|---" + "|---:" * len(AGGREGATORS) + "|"]
            for atk in ATTACKS:
                L.append(f"| {atk} | " + " | ".join(fmt(by(final, 'E4', f'{a}/{p}/{atk}/{g}')) for g in AGGREGATORS) + " |")
            L.append("")
        L += ["File `paired_e4.csv` có thêm hai đại lượng paired kèm CI 95%: lợi ích so với equal_mean cùng tấn công, "
              "và net drop so với hệ thống sạch không phòng thủ (none/equal_mean). Net drop là thứ người dùng thật sự chịu.", ""]
    simple("E4b", "E4b – Khai gian số mẫu khi server dùng weighted mean")

    if any(r["experiment"] == "E5" for r in final):
        L += ["## E5 – DP-FedAvg (client-level, mọi client tham gia)", "",
              "| σ | Test acc (%) | ε (δ = 1e-5) |", "|---|---:|---:|"]
        for s in cfg.dp_sigmas:
            L.append(f"| {s} | {fmt(by(final, 'E5', f'sigma={s}'))} | {eps_rdp(s, cfg.rounds):.1f} |")
        L += ["", f"Clip C = {cfg.dp_clip} cố định (chưa dùng adaptive clipping). ε tính bằng RDP composition không có "
              "khuếch đại do subsampling, nên là cận trên thô.", ""]

    if leaks:
        L += ["## E6 – Rò rỉ thông tin từ update (softmax regression)", "",
              f"- Khôi phục ảnh từ gradient batch=1: sai số lớn nhất {max(l['img_err'] for l in leaks):.1e}, "
              f"đoán đúng nhãn {100 * np.mean([l['label_acc'] for l in leaks]):.0f}%.",
              f"- Phân bố nhãn của client từ gradient FedSGD vòng 1 (alpha=0.1): sai số L1 lớn nhất {max(l['hist_l1'] for l in leaks):.1e}.",
              f"- Từ bias của update FedAvg ({cfg.local_epochs} epoch): Spearman trung bình với phân bố nhãn thật "
              f"{np.mean([l['rho_fedavg'] for l in leaks]):.2f}.", ""]

    L += ["## Giới hạn", "",
          "- Dữ liệu nhỏ, ảnh 8x8, 7 client. K nhỏ khiến DP và partial participation chỉ mang tính minh họa.",
          "- Hyperparameter nền (lr cục bộ 0.1, lr FedSGD 0.5, 2 epoch) kế thừa từ bản đầu, chỉ E3 được tune trên validation.",
          "- Kẻ tấn công ALIE/IPM được giả định biết update của client trung thực; chưa có tấn công thích ứng "
          "tối ưu riêng cho từng cách gộp, nên robustness ở đây là cận trên lạc quan.",
          "- Server biết đúng f = f_assumed. Nếu f thật > f_assumed, trimmed và Krum không còn bảo đảm.",
          f"- {n} seed: chênh lệch nằm trong CI 95% không nên diễn giải là khác biệt thật.", ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def plots(cfg: Config, final, curves, part_rows, leaks, out: Path) -> None:
    def mean_curve(exp, setting):
        rows = [r for r in curves if r["experiment"] == exp and r["setting"] == setting]
        seeds = sorted({r["seed"] for r in rows})
        return np.mean([[r["test_acc"] for r in rows if r["seed"] == s] for s in seeds], axis=0) * 100

    def line_fig(exp, title, name):
        settings = list(dict.fromkeys(r["setting"] for r in final if r["experiment"] == exp))
        if not settings:
            return
        plt.figure(figsize=(8, 4.6))
        for s in settings:
            plt.plot(range(1, cfg.rounds + 1), mean_curve(exp, s), label=s, lw=1.6)
        plt.xlabel("Vòng giao tiếp" + (" (central: cùng sample exposure)" if exp == "E1" else ""))
        plt.ylabel("Test accuracy (%)"); plt.ylim(0, 100); plt.title(title)
        plt.grid(alpha=0.25); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(out / name, dpi=150); plt.close()

    line_fig("E1", "E1 – IID sạch", "fig_e1_curves.png")
    line_fig("E3", "E3 – non-IID alpha=0.1, E=20 (cấu hình chọn trên validation)", "fig_e3_drift.png")

    if any(r["experiment"] == "E2" for r in final):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for p, col in zip(("IID", "alpha=0.5", "alpha=0.1"), ("#1D9E75", "#D9A21B", "#C0392B")):
            Es = list(cfg.drift_epochs)
            ax.plot(Es, [100 * np.mean(by(final, "E2", f"FedAvg/{p}/E={E}")) for E in Es], "o-", color=col, label=f"{p}: toàn cục")
            ax.plot(Es, [100 * np.mean(by(final, "E2", f"FedAvg/{p}/E={E}", "worst_client_acc")) for E in Es], "o--",
                    color=col, alpha=0.7, label=f"{p}: client tệ nhất")
        ax.set_xscale("log"); ax.set_xticks(cfg.drift_epochs, [str(e) for e in cfg.drift_epochs])
        ax.set_xlabel("Số epoch cục bộ mỗi vòng"); ax.set_ylabel("Test accuracy (%)")
        ax.set_title("E2 – FedAvg: client drift"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_e2_drift.png", dpi=150); plt.close(fig)

    if any(r["experiment"] == "E4" for r in final):
        combos = [(a, p) for a in ("FedSGD", "FedAvg") for p in ("IID", "alpha=0.5", "alpha=0.1")]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, (a, p) in zip(axes.ravel(), combos):
            M = np.array([[100 * np.mean(by(final, "E4", f"{a}/{p}/{atk}/{g}")) for g in AGGREGATORS] for atk in ATTACKS])
            im = ax.imshow(M, vmin=0, vmax=100, cmap="RdYlGn")
            ax.set_xticks(range(len(AGGREGATORS)), AGGREGATORS, fontsize=8)
            ax.set_yticks(range(len(ATTACKS)), ATTACKS, fontsize=8)
            for i in range(len(ATTACKS)):
                for j in range(len(AGGREGATORS)):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"{a}, {p}", fontsize=10)
        fig.colorbar(im, ax=axes, label="Test accuracy (%)", shrink=0.7)
        fig.suptitle(f"E4 – {cfg.f_attack}/{cfg.n_clients} client độc hại, server giả định f={cfg.f_assumed}")
        fig.savefig(out / "fig_e4_attack.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    if any(r["experiment"] == "E5" for r in final):
        plt.figure(figsize=(6, 4))
        acc = [100 * np.mean(by(final, "E5", f"sigma={s}")) for s in cfg.dp_sigmas]
        plt.plot(cfg.dp_sigmas, acc, "o-")
        for s, a in zip(cfg.dp_sigmas, acc):
            plt.annotate("không DP" if s == 0 else f"ε≈{eps_rdp(s, cfg.rounds):.0f}", (s, a),
                         textcoords="offset points", xytext=(5, 6))
        plt.xlabel("Noise multiplier σ"); plt.ylabel("Test accuracy (%)"); plt.ylim(0, 100)
        plt.title(f"E5 – DP-FedAvg, {cfg.n_clients} client, δ=1e-5"); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(out / "fig_e5_dp.png", dpi=150); plt.close()

    if leaks:
        fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
        for j, (orig, rec, y) in enumerate(leaks[0]["imgs"]):
            axes[0, j].imshow(orig.reshape(8, 8), cmap="gray_r"); axes[0, j].set_title(f"gốc ({y})", fontsize=8)
            axes[1, j].imshow(rec.reshape(8, 8), cmap="gray_r"); axes[1, j].set_title("từ gradient", fontsize=8)
        for a in axes.ravel():
            a.axis("off")
        fig.tight_layout(); fig.savefig(out / "fig_e6_leak.png", dpi=150); plt.close(fig)

    s0 = cfg.seeds[0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, p in zip(axes, ("IID", "alpha=0.5", "alpha=0.1")):
        rows = sorted((r for r in part_rows if r["seed"] == s0 and r["partition"] == p), key=lambda r: r["client_id"])
        C = np.array([json.loads(r["class_counts"]) for r in rows], float)
        im = ax.imshow(C / C.sum(1, keepdims=True), vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_title(f"{p} (seed {s0})"); ax.set_xlabel("Lớp"); ax.set_xticks(range(10))
    axes[0].set_ylabel("Client")
    fig.colorbar(im, ax=axes, label="Tỷ lệ lớp trong client", shrink=0.8)
    fig.savefig(out / "fig_partitions.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="Smoke test: 2 seed, 20 vòng")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seeds", type=str, help="Ví dụ 0,1,2,3,4")
    ap.add_argument("--clients", type=int, default=7)
    ap.add_argument("--model", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--only", type=str, help="Ví dụ E1,E4")
    ap.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    ap.add_argument("--out", type=str)
    a = ap.parse_args()
    cfg = Config(rounds=a.rounds or (20 if a.quick else 100),
                 seeds=tuple(int(x) for x in a.seeds.split(",")) if a.seeds else ((0, 1) if a.quick else (0, 1, 2, 3, 4)),
                 n_clients=a.clients, model=a.model, workers=a.workers,
                 only=tuple(x.strip() for x in a.only.split(",")) if a.only else Config.only,
                 out=a.out or f"results_final_{a.model}{'_quick' if a.quick else ''}")
    if cfg.n_clients < 2 * cfg.f_assumed + 1 or cfg.n_clients < cfg.f_assumed + 3:
        raise ValueError("Số client quá ít so với f_assumed.")
    out = Path(cfg.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    jobs = [(cfg, s) for s in cfg.seeds]
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as ex:
            results = list(ex.map(run_seed, jobs))
    else:
        results = [run_seed(j) for j in jobs]

    final = [r for res in results for r in res["final"]]
    curves = [r for res in results for r in res["curves"]]
    part_rows = [r for res in results for r in res["parts"]]
    checks = [r for res in results for r in res["checks"]]
    leaks = [res["extra"]["leak"] for res in results if "leak" in res["extra"]]
    paired = paired_tables(final) if any(r["experiment"] == "E4" for r in final) else []

    write_csv(out / "results.csv", final)
    write_csv(out / "curves.csv", curves)
    write_csv(out / "partitions.csv", part_rows)
    write_csv(out / "self_checks.csv", checks)
    write_csv(out / "paired_e4.csv", paired)
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "tuning": TUNING, "numpy": np.__version__,
                                                 "sklearn": sklearn.__version__, "n_params": Model(cfg.model, cfg.hidden).n_params},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    report(cfg, final, checks, paired, leaks, out)
    plots(cfg, final, curves, part_rows, leaks, out)
    print(f"Xong {len(final)} run trong {time.time() - t0:.0f}s. Kết quả ở {out}")


if __name__ == "__main__":
    main()

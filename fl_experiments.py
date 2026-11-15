"""
Thi nghiem FL mo rong tu fl_simple.py. Chi can numpy, scikit-learn, matplotlib.

Chay:  python fl_experiments.py
Ket qua ghi vao thu muc results/: results.csv, summary.md, fig_curves.png, fig_attack.png

Bon thi nghiem (moi cau hinh chay 5 seed, bao cao trung binh +- do lech chuan):
  E0  Kiem tra: FedSGD tren du lieu IID bang gradient descent tap trung
  E1  FL sach: hoc tap trung (moc tran) so voi FedSGD va FedAvg
  E2  Du lieu lech (non-IID, Dirichlet alpha): FedSGD va FedAvg
  E3  Client doc hai (dao dau x5): trung binh thuong va trimmed mean
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

K = 7                      # so client
ROUNDS = 100               # so vong giao tiep
SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------- Mo hinh: softmax regression (64 dac trung, 10 lop, d = 650) ----------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(w, X):
    return softmax(X @ w[:640].reshape(64, 10) + w[640:])


def accuracy(w, X, y):
    return float((predict_proba(w, X).argmax(axis=1) == y).mean())


def gradient(w, X, y):
    p = predict_proba(w, X)
    p[np.arange(len(y)), y] -= 1.0
    return np.concatenate([(X.T @ p / len(y)).ravel(), p.mean(axis=0)])


# ---------- Chia du lieu cho client ----------
def split_iid(y, rng):
    return np.array_split(rng.permutation(len(y)), K)


def split_dirichlet(y, alpha, rng, min_size=10):
    """Voi moi lop, chia cho K client theo ty le lay tu Dirichlet(alpha). alpha nho = rat lech."""
    while True:
        parts = [[] for _ in range(K)]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            q = rng.dirichlet(alpha * np.ones(K))
            cuts = (np.cumsum(q) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k] += chunk.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [np.array(p) for p in parts]


# ---------- Cap nhat cua client: tra ve "gradient gia" de server dung chung mot cach gop ----------
def update_fedsgd(w, X, y, rng):
    return gradient(w, X, y)


def update_fedavg(w, X, y, rng, epochs=2, batch=32, lr=0.1):
    wk = w.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            b = order[i:i + batch]
            wk -= lr * gradient(wk, X[b], y[b])
    return w - wk


# ---------- Gop o server ----------
def aggregate(G, n, mode="mean", k=0):
    if mode == "mean":
        return (n / n.sum()) @ G
    S = np.sort(G, axis=0)                      # trimmed mean theo tung tham so
    return S[k:G.shape[0] - k].mean(axis=0)


# ---------- Mot lan chay FL ----------
def run_fl(data, parts, algo, f=0, agg="mean", k=0, rng=None):
    Xtr, ytr, Xte, yte = data
    clients = [(Xtr[p], ytr[p]) for p in parts]
    n = np.array([len(p) for p in parts], dtype=float)
    upd = update_fedsgd if algo == "fedsgd" else update_fedavg
    eta = 0.5 if algo == "fedsgd" else 1.0
    w, curve = np.zeros(650), []
    for _ in range(ROUNDS):
        ups = [upd(w, Xc, yc, rng) for Xc, yc in clients]
        for i in range(f):                      # client doc hai: dao dau x5
            ups[i] = -5.0 * ups[i]
        w = w - eta * aggregate(np.stack(ups), n, agg, k)
        curve.append(accuracy(w, Xte, yte))
    return curve


# ---------- Hoc tap trung ----------
def run_central_gd(data):
    """Gradient descent tren toan bo du lieu, cung so buoc va learning rate voi FedSGD."""
    Xtr, ytr, Xte, yte = data
    w = np.zeros(650)
    for _ in range(ROUNDS):
        w = w - 0.5 * gradient(w, Xtr, ytr)
    return accuracy(w, Xte, yte)


def run_central_sgd(data, rng, epochs=100, batch=32, lr=0.1):
    """Moc tran: SGD mini-batch nhieu epoch tren toan bo du lieu."""
    Xtr, ytr, Xte, yte = data
    w = np.zeros(650)
    for _ in range(epochs):
        order = rng.permutation(len(ytr))
        for i in range(0, len(ytr), batch):
            b = order[i:i + batch]
            w -= lr * gradient(w, Xtr[b], ytr[b])
    return accuracy(w, Xte, yte)


def load(seed):
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.25, random_state=seed)
    return Xtr, ytr, Xte, yte


def ms(v):
    return f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, curves, res = [], {}, {}

    def log(exp, setting, seed, acc):
        rows.append([exp, setting, seed, round(acc, 4)])
        res.setdefault((exp, setting), []).append(acc)

    alphas = [("IID", None), ("alpha=1.0", 1.0), ("alpha=0.5", 0.5), ("alpha=0.1", 0.1)]
    attack_cfgs = [("f=0, trung bình", 0, "mean", 0), ("f=1, trung bình", 1, "mean", 0),
                   ("f=2, trung bình", 2, "mean", 0), ("f=1, trimmed k=1", 1, "trimmed", 1),
                   ("f=2, trimmed k=1 (cắt thiếu)", 2, "trimmed", 1),
                   ("f=2, trimmed k=2", 2, "trimmed", 2),
                   ("f=0, trimmed k=2 (không bị tấn công)", 0, "trimmed", 2)]

    for seed in SEEDS:
        data = load(seed)
        ytr = data[1]
        rng = np.random.default_rng(seed)
        iid = split_iid(ytr, rng)

        # E0 + E1: FL sach, du lieu IID
        c_sgd = run_fl(data, iid, "fedsgd", rng=rng)
        c_avg = run_fl(data, iid, "fedavg", rng=rng)
        curves.setdefault("FedSGD (IID)", []).append(c_sgd)
        curves.setdefault("FedAvg (IID)", []).append(c_avg)
        log("E0", "central GD", seed, run_central_gd(data))
        log("E0", "FedSGD IID", seed, c_sgd[-1])
        log("E1", "hoc tap trung (SGD, moc tran)", seed, run_central_sgd(data, rng))
        log("E1", "FedSGD", seed, c_sgd[-1])
        log("E1", "FedAvg", seed, c_avg[-1])

        # E2: du lieu lech
        for name, alpha in alphas:
            parts = iid if alpha is None else split_dirichlet(ytr, alpha, rng)
            for algo in ("fedsgd", "fedavg"):
                curve = c_sgd if (alpha is None and algo == "fedsgd") else \
                    c_avg if (alpha is None and algo == "fedavg") else run_fl(data, parts, algo, rng=rng)
                log("E2", f"{algo} ({name})", seed, curve[-1])
                if alpha == 0.1 and algo == "fedavg":
                    curves.setdefault("FedAvg (alpha=0.1)", []).append(curve)

        # E3: client doc hai (FedSGD, du lieu IID)
        for name, f, agg, k in attack_cfgs:
            log("E3", name, seed, run_fl(data, iid, "fedsgd", f=f, agg=agg, k=k, rng=rng)[-1])

    # ---- Ghi CSV va bang tom tat ----
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["thi_nghiem", "cau_hinh", "seed", "accuracy_cuoi"])
        wr.writerows(rows)

    lines = [f"Ket qua: accuracy cuoi tren tap test (%), trung binh ± do lech chuan qua {len(SEEDS)} seed",
             f"K = {K} client, {ROUNDS} vong\n"]
    for exp in ("E0", "E1", "E2", "E3"):
        lines.append(f"## {exp}")
        lines.append("| Cau hinh | Accuracy |\n|---|---|")
        for (e, s), v in res.items():
            if e == exp:
                lines.append(f"| {s} | {ms(v)} |")
        lines.append("")
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write(summary)
    print(summary)

    # ---- Hinh 1: accuracy theo vong ----
    plt.figure(figsize=(7, 4))
    for name, cs in curves.items():
        plt.plot(range(1, ROUNDS + 1), 100 * np.mean(cs, axis=0), label=name)
    plt.xlabel("Vòng"); plt.ylabel("Accuracy trên tập test (%)")
    plt.title(f"FL sạch: FedSGD và FedAvg ({K} client, trung bình {len(SEEDS)} seed)")
    plt.ylim(0, 100); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_curves.png"), dpi=150); plt.close()

    # ---- Hinh 2: client doc hai ----
    names = [c[0] for c in attack_cfgs]
    means = [100 * np.mean(res[("E3", n)]) for n in names]
    stds = [100 * np.std(res[("E3", n)]) for n in names]
    plt.figure(figsize=(8, 4.5))
    plt.barh(range(len(names)), means, xerr=stds, color="#1D9E75")
    plt.yticks(range(len(names)), names); plt.gca().invert_yaxis()
    plt.xlabel("Accuracy trên tập test (%)"); plt.xlim(0, 100)
    plt.title("Client độc hại (đảo dấu ×5) và phòng thủ trimmed mean")
    plt.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig_attack.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
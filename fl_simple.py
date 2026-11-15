"""
FL don gian: softmax regression + FedSGD tren anh chu so 8x8 (scikit-learn).
Chi can numpy va scikit-learn. Chay thu:

    python fl_simple.py                         # FL sach, khong tan cong
    python fl_simple.py --f 1                   # 1 client doc hai, trung binh thuong
    python fl_simple.py --f 2 --agg trimmed --k 2   # 2 doc hai, trimmed mean k = 2
"""
import argparse
import hashlib

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ---------- 1. Mo hinh: softmax regression (64 dac trung, 10 lop, d = 650) ----------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def unpack(w):
    return w[:640].reshape(64, 10), w[640:]          # W (64x10), b (10)


def predict_proba(w, X):
    W, b = unpack(w)
    return softmax(X @ W + b)


def accuracy(w, X, y):
    return float((predict_proba(w, X).argmax(axis=1) == y).mean())


# ---------- 2. Gradient cua mot client (tinh tren du lieu rieng) ----------
def gradient(w, X, y):
    p = predict_proba(w, X)
    p[np.arange(len(y)), y] -= 1.0                   # p - one_hot(y)
    gW = X.T @ p / len(y)
    gb = p.mean(axis=0)
    return np.concatenate([gW.ravel(), gb])


# ---------- 3. Cach gop tai server ----------
def aggregate(G, n, mode="mean", k=0):
    """G: (K, d) cac gradient; n: (K,) so mau moi client."""
    if mode == "mean":                               # trung binh co trong so n_k / N
        return (n / n.sum()) @ G
    if mode == "trimmed":                            # bo k nho nhat, k lon nhat tung tham so
        assert G.shape[0] > 2 * k, "can K > 2k"
        S = np.sort(G, axis=0)
        return S[k:G.shape[0] - k].mean(axis=0)
    raise ValueError(mode)


# ---------- 4. (Noi voi blockchain) ban cam ket hash cua mot cap nhat ----------
def serialize(vec):
    return np.ascontiguousarray(vec, dtype="<f4").tobytes()   # float32, little-endian


def commit_hash(client_id, rnd, version, g):
    head = f"{client_id}|{rnd}|{version}|".encode()
    return hashlib.sha256(head + serialize(g)).hexdigest()


# ---------- 5. Vong lap FL ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--f", type=int, default=0, help="so client doc hai (dao dau x5)")
    ap.add_argument("--agg", choices=["mean", "trimmed"], default="mean")
    ap.add_argument("--k", type=int, default=0, help="k cua trimmed mean (nen >= f)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # Du lieu: chia train/test, roi chia train cho cac client (IID)
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=a.seed)
    rng = np.random.default_rng(a.seed)
    parts = np.array_split(rng.permutation(len(Xtr)), a.clients)
    clients = [(Xtr[i], ytr[i]) for i in parts]
    n = np.array([len(c[1]) for c in clients], dtype=float)

    w = np.zeros(650)                                # mo hinh toan cuc, bat dau tu 0
    print(f"K={a.clients}, moi client ~{int(n.mean())} anh, f={a.f}, agg={a.agg}, k={a.k}")

    for t in range(1, a.rounds + 1):
        # Buoc 1-3: server gui w; moi client tinh gradient tren du lieu rieng va gui ve
        grads = [gradient(w, Xc, yc) for Xc, yc in clients]
        for i in range(a.f):                         # client doc hai: dao dau roi nhan 5
            grads[i] = -5.0 * grads[i]
        G = np.stack(grads)

        if t == 1:                                   # vi du: hash cam ket cua client 0, vong 1
            print("hash cam ket client 0, vong 1:", commit_hash(0, t, 0, G[0])[:16], "...")

        # Buoc 4-5: server gop va cap nhat mo hinh
        w = w - a.eta * aggregate(G, n, a.agg, a.k)

        if t % 10 == 0 or t == 1:
            print(f"vong {t:3d}: accuracy test = {accuracy(w, Xte, yte) * 100:5.1f}%")


if __name__ == "__main__":
    main()
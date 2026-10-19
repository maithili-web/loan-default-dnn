"""
Loan Default Prediction with a Deep Neural Network (built from scratch in NumPy)
=================================================================================
Mini project: predict whether a borrower will default on a loan.

Pipeline
  1. Generate a realistic synthetic loan dataset (no real customer data is used)
  2. Preprocess (log transforms, one-hot encoding, standardisation)
  3. Train a 3-hidden-layer MLP: forward/backward pass, ReLU, dropout, L2, Adam,
     class-weighted loss for imbalance, early stopping
  4. Verify backprop with a numerical gradient check
  5. Evaluate vs. a Logistic Regression baseline (ROC-AUC, PR-AUC, F1, confusion matrix)
  6. Explain the model with permutation feature importance

Run:  python loan_default_dnn.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def make_dataset(n=20000, seed=SEED, target_rate=0.15):
    """Synthetic loan book. Default risk depends non-linearly on the features."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, n)
    income = np.exp(rng.normal(13.3, 0.5, n))                      # annual income (INR)
    credit_score = np.clip(rng.normal(690, 80, n), 300, 900).round()  # CIBIL-style 300-900
    employment_years = np.clip(age - 21 - rng.exponential(3, n), 0, 40).round(1)
    num_credit_lines = rng.poisson(4, n)
    past_delinquencies = rng.poisson(0.6, n)
    loan_term = rng.choice([12, 24, 36, 48, 60], n)
    loan_amount = (income * rng.uniform(0.2, 2.0, n)).round(-3)
    home = rng.choice(["rent", "own", "mortgage"], n, p=[0.45, 0.25, 0.30])
    purpose = rng.choice(["personal", "education", "home_improvement", "business", "medical"],
                         n, p=[0.35, 0.15, 0.20, 0.15, 0.15])

    emi_ratio = (loan_amount / loan_term) / (income / 12)          # monthly EMI / monthly income

    logit = (
        -0.018 * (credit_score - 690)                              # better score -> safer
        + 4.0 * emi_ratio                                          # heavy EMI burden -> riskier
        + 0.45 * past_delinquencies
        + 0.012 * past_delinquencies * np.maximum(0, 650 - credit_score) / 10  # interaction
        - 0.9 * np.tanh(employment_years / 8)                      # stability saturates
        + 0.08 * np.maximum(0, num_credit_lines - 6) ** 1.5        # over-leverage
        + 0.5 * (purpose == "business") + 0.25 * (purpose == "medical")
        - 0.35 * (home == "own") + 0.2 * (home == "rent")
        + 0.006 * (loan_term - 36)
        + rng.normal(0, 0.6, n)                                    # unobserved noise
    )
    # shift the intercept so the overall default rate hits the target
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    p_default = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
    default = (rng.uniform(size=n) < p_default).astype(int)

    return pd.DataFrame(dict(
        age=age, annual_income=income.round(-2), credit_score=credit_score,
        employment_years=employment_years, num_credit_lines=num_credit_lines,
        past_delinquencies=past_delinquencies, loan_term_months=loan_term,
        loan_amount=loan_amount, emi_to_income=emi_ratio.round(4),
        home_ownership=home, loan_purpose=purpose, default=default))


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
class Preprocessor:
    LOG_COLS = ["annual_income", "loan_amount"]
    CAT_COLS = ["home_ownership", "loan_purpose"]

    def fit(self, df):
        X = self._raw(df)
        self.columns = X.columns.tolist()
        self.mean, self.std = X.mean(), X.std().replace(0, 1)
        return self

    def _raw(self, df):
        X = df.drop(columns="default").copy()
        X[self.LOG_COLS] = np.log1p(X[self.LOG_COLS])
        X = pd.get_dummies(X, columns=self.CAT_COLS, dtype=float)
        if hasattr(self, "columns"):
            X = X.reindex(columns=self.columns, fill_value=0.0)
        return X

    def transform(self, df):
        return ((self._raw(df) - self.mean) / self.std).to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# 3. Deep neural network from scratch
# --------------------------------------------------------------------------- #
class DeepNet:
    """MLP: [input] -> ReLU(+dropout) x hidden -> sigmoid output."""

    def __init__(self, sizes, dropout=0.2, l2=1e-4, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.dropout, self.l2 = dropout, l2
        self.W = [self.rng.normal(0, np.sqrt(2 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]  # He init
        self.b = [np.zeros(n) for n in sizes[1:]]
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def forward(self, X, training=False):
        inputs, zs, masks = [], [], []
        a = X
        for l in range(len(self.W) - 1):
            inputs.append(a)
            z = a @ self.W[l] + self.b[l]
            zs.append(z)
            a = np.maximum(0, z)
            if training and self.dropout > 0:
                keep = 1 - self.dropout
                mask = (self.rng.uniform(size=a.shape) < keep) / keep   # inverted dropout
                a = a * mask
            else:
                mask = np.ones_like(a)
            masks.append(mask)
        inputs.append(a)
        logits = (a @ self.W[-1] + self.b[-1]).ravel()
        return logits, (inputs, zs, masks)

    def predict_proba(self, X):
        return self._sigmoid(self.forward(X)[0])

    def loss(self, X, y, pos_weight=1.0, training=False):
        """Weighted binary cross-entropy (+ L2 penalty)."""
        z, _ = self.forward(X, training)
        # numerically stable BCE with logits
        bce = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        w = np.where(y == 1, pos_weight, 1.0)
        reg = 0.5 * self.l2 * sum((w_ ** 2).sum() for w_ in self.W)
        return (w * bce).mean() + reg

    def gradients(self, X, y, pos_weight=1.0):
        z, (inputs, zs, masks) = self.forward(X, training=True)
        p = self._sigmoid(z)
        w = np.where(y == 1, pos_weight, 1.0)
        da = ((p - y) * w / len(y))[:, None]                 # dLoss/dlogits
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = inputs[l].T @ da + self.l2 * self.W[l]
            gb[l] = da.sum(0)
            if l > 0:
                dh = (da @ self.W[l].T) * masks[l - 1]       # through dropout
                da = dh * (zs[l - 1] > 0)                    # through ReLU
        return gW, gb

    def adam_step(self, gW, gb, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for params, grads, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for i in range(len(params)):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
                mhat, vhat = m[i] / (1 - b1 ** self.t), v[i] / (1 - b2 ** self.t)
                params[i] -= lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, Xtr, ytr, Xva, yva, epochs=120, batch=256, lr=1e-3, patience=12, verbose=True):
        pos_weight = (ytr == 0).sum() / (ytr == 1).sum()      # handle class imbalance
        history = {"train": [], "val": []}
        best, best_state, wait = np.inf, None, 0
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(len(ytr))
            for s in range(0, len(idx), batch):
                b = idx[s:s + batch]
                gW, gb = self.gradients(Xtr[b], ytr[b], pos_weight)
                self.adam_step(gW, gb, lr)
            tr = self.loss(Xtr, ytr, pos_weight)
            va = self.loss(Xva, yva, pos_weight)
            history["train"].append(tr)
            history["val"].append(va)
            if va < best - 1e-5:
                best, wait = va, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"epoch {ep:3d}  train loss {tr:.4f}  val loss {va:.4f}")
            if wait >= patience:
                if verbose:
                    print(f"early stopping at epoch {ep} (best val loss {best:.4f})")
                break
        self.W, self.b = best_state
        history["best_epoch"] = int(np.argmin(history["val"]) + 1)
        return history


def gradient_check(seed=0):
    """Compare analytic backprop gradients against finite differences."""
    rng = np.random.default_rng(seed)
    net = DeepNet([6, 5, 4, 1], dropout=0.0, l2=1e-3, seed=seed)
    X, y = rng.normal(size=(20, 6)), (rng.uniform(size=20) < 0.3).astype(float)
    gW, _ = net.gradients(X, y, pos_weight=3.0)
    worst, eps = 0.0, 1e-5
    for l in range(len(net.W)):
        for _ in range(8):
            i, j = rng.integers(net.W[l].shape[0]), rng.integers(net.W[l].shape[1])
            old = net.W[l][i, j]
            net.W[l][i, j] = old + eps
            up = net.loss(X, y, 3.0)
            net.W[l][i, j] = old - eps
            dn = net.loss(X, y, 3.0)
            net.W[l][i, j] = old
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - gW[l][i, j]) / max(1e-8, abs(num) + abs(gW[l][i, j])))
    return worst


# --------------------------------------------------------------------------- #
# 4. Evaluation helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                threshold=thr, precision=prec, recall=rec, f1=f1,
                confusion_matrix=confusion_matrix(y, pred).tolist())


def permutation_importance(net, prep, df_test, y, n_repeats=5, seed=SEED):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, net.predict_proba(prep.transform(df_test)))
    feats = [c for c in df_test.columns if c != "default"]
    out = {}
    for f in feats:
        drops = []
        for _ in range(n_repeats):
            d = df_test.copy()
            d[f] = rng.permutation(d[f].to_numpy())
            drops.append(base - roc_auc_score(y, net.predict_proba(prep.transform(d))))
        out[f] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    print("== Gradient check ==")
    err = gradient_check()
    print(f"max relative error analytic vs numeric gradient: {err:.2e}  ({'PASS' if err < 1e-5 else 'FAIL'})")

    df = make_dataset()
    df.to_csv(os.path.join(OUT_DIR, "loan_data.csv"), index=False)
    print(f"\n== Data ==\n{len(df)} loans, default rate {df['default'].mean():.1%}")

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["default"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["default"], random_state=SEED)
    prep = Preprocessor().fit(train_df)
    Xtr, Xva, Xte = (prep.transform(d) for d in (train_df, val_df, test_df))
    ytr, yva, yte = (d["default"].to_numpy(float) for d in (train_df, val_df, test_df))
    print(f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)}; {Xtr.shape[1]} input features")

    print("\n== Training DNN ==")
    net = DeepNet([Xtr.shape[1], 64, 32, 16, 1], dropout=0.2, l2=1e-4)
    hist = net.fit(Xtr, ytr, Xva, yva)

    print("\n== Evaluation (test set) ==")
    p_dnn_va, p_dnn = net.predict_proba(Xva), net.predict_proba(Xte)
    thr_dnn = best_f1_threshold(yva, p_dnn_va)
    res_dnn = evaluate(yte, p_dnn, thr_dnn)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    p_lr_va, p_lr = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]
    res_lr = evaluate(yte, p_lr, best_f1_threshold(yva, p_lr_va))

    for name, r in (("Deep NN", res_dnn), ("Logistic Regression", res_lr)):
        print(f"{name:20s} ROC-AUC {r['roc_auc']:.3f} | PR-AUC {r['pr_auc']:.3f} | "
              f"P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} @thr {r['threshold']:.2f}")

    imp = permutation_importance(net, prep, test_df, yte)

    # ---- plots ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(hist["train"], label="train")
    ax[0, 0].plot(hist["val"], label="validation")
    ax[0, 0].axvline(hist["best_epoch"] - 1, ls="--", c="grey", lw=1, label=f"best epoch {hist['best_epoch']}")
    ax[0, 0].set(title="Training curve (weighted BCE)", xlabel="epoch", ylabel="loss")
    ax[0, 0].legend()

    for name, p in (("Deep NN", p_dnn), ("Logistic Reg.", p_lr)):
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0, 1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0, 1].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0, 1].legend()

    cm = np.array(res_dnn["confusion_matrix"])
    ax[1, 0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1, 0].text(j, i, str(v), ha="center", va="center",
                      color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax[1, 0].set(title=f"DNN confusion matrix (threshold {thr_dnn:.2f})", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["Repaid", "Default"], yticklabels=["Repaid", "Default"],
                 xlabel="Predicted", ylabel="Actual")

    top = list(imp.items())[:8][::-1]
    ax[1, 1].barh([k for k, _ in top], [v for _, v in top], color="#3b7dd8")
    ax[1, 1].set(title="Permutation importance (drop in test AUC)", xlabel="AUC decrease")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "results.png"), dpi=130)

    out = dict(gradient_check_max_rel_error=err, default_rate=float(df["default"].mean()),
               best_epoch=hist["best_epoch"], dnn=res_dnn, logistic_regression=res_lr,
               permutation_importance=imp)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results.png, metrics.json, loan_data.csv")


if __name__ == "__main__":
    main()

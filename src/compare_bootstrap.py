"""
Compare two systems on the SAME test articles with a paired bootstrap, and print
per-class F1 for both. Use it to answer: "is the Transformer really better (or worse)
than the TF-IDF baseline, or is the gap just noise from a tiny test set?"

Inputs are prediction CSVs with columns: label_name, pred (and optionally text)
(rows are in the same order as data/processed/raw_test.csv; a 'text' column is optional).

Usage:
  python src/compare_bootstrap.py \
      --a results/preds_tfidf_char_svm_raw.csv --a_name "TF-IDF char SVM" \
      --b results/transformers/preds_E3_afroxlmr_base_len256_wloss_s44.csv --b_name "AfroXLMR E3"

Paired bootstrap idea: resample the test articles WITH replacement 2000 times; on each
resample compute macro-F1 for both systems and take the difference. The 95% interval of
that difference tells us whether 0 (no difference) is plausible.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data_prep import CLASS_NAMES


def macro_f1(y, p):
    return f1_score(y, p, labels=CLASS_NAMES, average="macro", zero_division=0)


def main(a):
    A, B = pd.read_csv(a.a), pd.read_csv(a.b)
    assert len(A) == len(B), "prediction files have different lengths"
    # sanity check: the rows must be the same articles in the same order
    if "text" in A.columns and "text" in B.columns:   # alignment check when excerpts are present
        assert (A.text.str.slice(0, 80) == B.text.str.slice(0, 80)).all(), "test rows are not aligned"
    assert (A.label_name == B.label_name).all()
    y, pa, pb = A.label_name.values, A.pred.values, B.pred.values

    fa, fb = macro_f1(y, pa), macro_f1(y, pb)
    print(f"macro-F1  {a.a_name}: {fa:.4f}   {a.b_name}: {fb:.4f}   difference (B-A): {fb - fa:+.4f}")
    print(f"accuracy  {a.a_name}: {(y == pa).mean():.4f}   {a.b_name}: {(y == pb).mean():.4f}")

    rng = np.random.default_rng(42)
    n, diffs, fa_s, fb_s = len(y), [], [], []
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, n)
        x, u, w = y[idx], pa[idx], pb[idx]
        sa, sb = macro_f1(x, u), macro_f1(x, w)
        fa_s.append(sa); fb_s.append(sb); diffs.append(sb - sa)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    ci = lambda v: tuple(float(x) for x in np.round(np.percentile(v, [2.5, 97.5]), 3))
    print(f"95% bootstrap CI macro-F1  {a.a_name}: {ci(fa_s)}   {a.b_name}: {ci(fb_s)}")
    print(f"95% bootstrap CI of the difference (B-A): [{lo:+.3f}, {hi:+.3f}]  "
          f"-> {'0 is inside: no reliable difference' if lo <= 0 <= hi else 'excludes 0: reliable difference'}")
    print(f"P(B better than A) over resamples: {(np.array(diffs) > 0).mean():.2f}")

    rows = []
    for c in CLASS_NAMES:
        rows.append({"class": c, "support": int((y == c).sum()),
                     f"F1 {a.a_name}": round(f1_score(y == c, pa == c, zero_division=0), 3),
                     f"F1 {a.b_name}": round(f1_score(y == c, pb == c, zero_division=0), 3)})
    print("\nPer-class F1 on the test split:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--a_name", default="A")
    p.add_argument("--b", required=True); p.add_argument("--b_name", default="B")
    p.add_argument("--n_boot", type=int, default=2000)
    main(p.parse_args())

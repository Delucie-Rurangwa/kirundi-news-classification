"""
Classical baselines for KIRNEWS: TF-IDF features + a linear classifier.

Why a TF-IDF baseline? It is fast, needs no GPU, and is the standard point
of comparison before fine-tuning a Transformer. If a fine-tuned model cannot
beat it, we must explain why.

Experiments run here:
  A. Clean split (deduplicated, our own 70/15/15 split): fit on train,
     evaluate on val and test.
  B. 5-fold stratified cross-validation over all deduplicated articles
     (more stable than one tiny test set, because rare classes have <10 items).
  C. "Leaky" official split: train on the authors' train.csv, test on their
     test.csv, to show how much duplicates inflate the scores.

Usage:
    python src/baseline_tfidf.py --version raw
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_prep import CLASS_NAMES, SEED


def make_model(kind: str) -> Pipeline:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           lowercase=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=150000,
                           sublinear_tf=True, lowercase=True)
    if kind == "tfidf_word_lr":
        return Pipeline([("tfidf", word),
                         ("clf", LogisticRegression(max_iter=2000, C=10,
                                                    class_weight="balanced"))])
    if kind == "tfidf_char_svm":
        return Pipeline([("tfidf", char),
                         ("clf", LinearSVC(C=1.0, class_weight="balanced"))])
    if kind == "tfidf_word+char_svm":
        return Pipeline([("tfidf", FeatureUnion([("w", word), ("c", char)])),
                         ("clf", LinearSVC(C=1.0, class_weight="balanced"))])
    raise ValueError(kind)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted",
                                          zero_division=0), 4)}


def main(args):
    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    v = args.version
    tr, va, te = (pd.read_csv(data / f"{v}_{s}.csv") for s in ("train", "val", "test"))
    otr, ote = (pd.read_csv(data / f"{v}_official_{s}.csv") for s in ("train", "test"))
    labels = list(range(len(CLASS_NAMES)))
    results = {}

    for kind in ("tfidf_word_lr", "tfidf_char_svm", "tfidf_word+char_svm"):
        r = {}
        # A. clean split
        m = make_model(kind).fit(tr.text, tr.y)
        r["clean_val"] = scores(va.y, m.predict(va.text))
        pred_te = m.predict(te.text)
        r["clean_test"] = scores(te.y, pred_te)
        # B. stratified 5-fold CV on all deduplicated articles
        allx = pd.concat([tr, va, te], ignore_index=True)
        f1s, accs = [], []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(allx.text, allx.y):
            mm = make_model(kind).fit(allx.text[a], allx.y[a])
            p = mm.predict(allx.text[b])
            s = scores(allx.y[b], p)
            f1s.append(s["macro_f1"]); accs.append(s["accuracy"])
        r["cv5_macro_f1_mean_std"] = [round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)]
        r["cv5_accuracy_mean_std"] = [round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4)]
        # C. leaky official split
        mo = make_model(kind).fit(otr.text, otr.y)
        r["official_leaky_test"] = scores(ote.y, mo.predict(ote.text))
        results[kind] = r
        print(kind, json.dumps(r))

        if kind == args.report_model:
            rep = classification_report(te.y, pred_te, labels=labels,
                                        target_names=CLASS_NAMES, zero_division=0,
                                        output_dict=True)
            results["per_class_report_" + kind] = rep
            cm = confusion_matrix(te.y, pred_te, labels=labels)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(labels); ax.set_yticks(labels)
            ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            for i in labels:
                for j in labels:
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion matrix - {kind} ({v}, clean test split)")
            fig.tight_layout()
            fig.savefig(out / f"confusion_{kind}_{v}.png", dpi=150)

            # save misclassified examples for the error analysis section
            err = te.assign(pred=[CLASS_NAMES[p] for p in pred_te])
            err = err[err.label_name != err.pred][["label_name", "pred", "text"]]
            err["text"] = err.text.str.slice(0, 300)
            err.to_csv(out / f"errors_{kind}_{v}.csv", index=False)

    (out / f"baseline_results_{v}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["raw", "cleaned"], default="raw")
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--report_model", default="tfidf_word+char_svm")
    main(p.parse_args())

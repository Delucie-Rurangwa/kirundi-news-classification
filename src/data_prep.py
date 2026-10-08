"""
Data preparation for the KIRNEWS Kirundi news-topic classification project.

What this script does (and why):
1. Loads the authors' official train/test CSVs (raw or cleaned version).
2. Maps the non-contiguous class IDs (1-7, 9, 11-14) to 0..11 using the
   class NAME, so the mapping cannot silently drift.
3. Measures train/test leakage in the OFFICIAL split (before any cleaning).
4. Merges official train+test, removes exact duplicates (after light
   normalisation) and drops articles whose identical text has conflicting labels.
5. Creates our own stratified train/validation/test split with a fixed seed.
6. Writes a JSON report with every count, so the Dataset section of the
   report can quote real numbers.

Usage:
    python src/data_prep.py --version raw
    python src/data_prep.py --version cleaned
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
# Label order follows classes.txt from the dataset authors.
CLASS_NAMES = ["politics", "sport", "economy", "health", "entertainment",
               "history", "technology", "culture", "religion",
               "environment", "education", "relationship"]
# Official numeric IDs -> class name (IDs 8 and 10 are unused in KIRNEWS).
ID_TO_NAME = {1: "politics", 2: "sport", 3: "economy", 4: "health",
              5: "entertainment", 6: "history", 7: "technology",
              9: "culture", 11: "religion", 12: "environment",
              13: "education", 14: "relationship"}
NAME_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
EN_LABEL_FIX = {"politic": "politics"}  # raw files spell it "politic"


def normalise(text: str) -> str:
    """Lowercase, unify apostrophes, collapse whitespace. Used ONLY to detect
    duplicates; the text we train on keeps its original form."""
    text = str(text).lower().replace("\u2019", "'").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_official(raw_dir: Path, version: str) -> pd.DataFrame:
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(raw_dir / version / f"{split}.csv")
        df["official_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["label_name"] = df["label"].map(ID_TO_NAME)
    if df["label_name"].isna().any():
        raise ValueError("Unexpected label id found in data")
    if "en_label" in df.columns:  # sanity check: numeric id agrees with name
        names = df["en_label"].replace(EN_LABEL_FIX)
        assert (names == df["label_name"]).all(), "label id/name mismatch"
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")
    return df


def add_text_and_y(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["text"] = d["title"].str.strip() + ". " + d["content"].str.strip()
    d["y"] = d["label_name"].map(NAME_TO_IDX)
    return d


def main(args):
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_official(raw_dir, args.version)
    report = {"version": args.version, "seed": SEED,
              "official_rows": int(len(df)),
              "official_train_rows": int((df.official_split == "train").sum()),
              "official_test_rows": int((df.official_split == "test").sum())}

    df["key"] = df["content"].map(normalise)

    # --- leakage in the OFFICIAL split (measured before any cleaning) -------
    tr_keys = set(df.loc[df.official_split == "train", "key"])
    te = df[df.official_split == "test"].drop_duplicates("key")
    report["official_test_unique"] = int(len(te))
    report["official_test_unique_also_in_train"] = int(te["key"].isin(tr_keys).sum())

    # --- exact duplicates and label conflicts -------------------------------
    n_labels = df.groupby("key")["label_name"].nunique()
    conflict_keys = set(n_labels[n_labels > 1].index)
    report["texts_with_conflicting_labels"] = len(conflict_keys)
    clean = df[~df["key"].isin(conflict_keys)]
    report["rows_dropped_conflicting"] = int(len(df) - len(clean))
    clean = clean.drop_duplicates("key").reset_index(drop=True)
    clean = add_text_and_y(clean)
    report["unique_clean_articles"] = int(len(clean))
    report["class_counts_after_dedup"] = clean["label_name"].value_counts().to_dict()

    # --- stratified 70/15/15 split ------------------------------------------
    train, rest = train_test_split(clean, test_size=0.30, stratify=clean["y"],
                                   random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["y"],
                                 random_state=SEED)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[["text", "y", "label_name"]].to_csv(
            out_dir / f"{args.version}_{name}.csv", index=False)
        report[f"{name}_size"] = int(len(part))
        report[f"{name}_class_counts"] = part["label_name"].value_counts().to_dict()

    # Keep the official split too, for the "leaky vs clean" experiment.
    for split in ("train", "test"):
        d = add_text_and_y(df[df.official_split == split])
        d[["text", "y", "label_name"]].to_csv(
            out_dir / f"{args.version}_official_{split}.csv", index=False)

    lens = clean["text"].str.split().str.len()
    report["words_per_article"] = {k: float(v) for k, v in
                                   lens.describe().round(1).to_dict().items()}
    (out_dir / f"{args.version}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["raw", "cleaned"], default="raw")
    p.add_argument("--raw_dir", default="data/raw/KIRNEWS")
    p.add_argument("--out_dir", default="data/processed")
    main(p.parse_args())

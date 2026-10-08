"""
Collect every results/transformers/result_*.json (one per run) plus the TF-IDF
baselines, and build the experiments table for the report:
mean +/- std over seeds for each configuration.

Usage:  python src/aggregate_results.py
"""
import json
from pathlib import Path

import pandas as pd

rows = []
for f in sorted(Path("results/transformers").glob("result_*.json")):
    r = json.loads(f.read_text())
    cfg = r["name"].rsplit("_s", 1)[0]          # strip the seed suffix
    rows.append({"config": cfg, "seed": r["seed"],
                 "val_macro_f1": r["val"]["macro_f1"],
                 "test_macro_f1": r["test"]["macro_f1"],
                 "test_accuracy": r["test"]["accuracy"]})
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No result files found in results/transformers/")
df.to_csv("results/all_runs.csv", index=False)

g = df.groupby("config")
table = g.agg(runs=("seed", "count"),
              val_macro_f1_mean=("val_macro_f1", "mean"),
              test_macro_f1_mean=("test_macro_f1", "mean"),
              test_macro_f1_std=("test_macro_f1", "std"),
              test_acc_mean=("test_accuracy", "mean")).round(4)
table.to_csv("results/experiments_table.csv")
print(table.to_string())
print("\nMarkdown version (paste into the report):\n")
print(table.reset_index().to_markdown(index=False))

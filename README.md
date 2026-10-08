# Kirundi News Topic Classification (KIRNEWS)

Individual NLP capstone: classify Kirundi news articles into 12 topics.

> Status: work in progress. Sections marked TODO are filled in as the project advances.

## Project links
- GitHub repository: TODO
- Live system (web app): TODO
- Demo video: TODO
- Report: TODO

## Problem
Automatically assign a Kirundi news article to one of 12 topics (politics, sport, economy, health,
entertainment, history, technology, culture, religion, environment, education, relationship).

## Dataset
KIRNEWS (Niyongabo et al., 2020, COLING): 4,612 articles from 8 Burundian news sources.
Download from the authors' release (see their GitHub: Andrews2017/KINNEWS-and-KIRNEWS-Corpus),
unzip into `data/raw/KIRNEWS/`. The raw data is not committed: article copyright belongs to the
original news sources.

Data-quality findings (measured by `src/data_prep.py`, see `data/processed/*_report.json`):
- Only 1,811 of the 4,612 rows are unique texts; every unique raw test article also appears in train.
- 35 texts carry conflicting labels (one election article appears 172 times under 3 labels).
- After deduplication and removing conflicts: 1,776 articles, strongly imbalanced
  (politics 611 ... history 9).

## How to run
```bash
pip install -r requirements.txt
python src/data_prep.py --version raw
python src/baseline_tfidf.py --version raw
```

## Repository layout
```
src/data_prep.py        dedupe, label remap, stratified split, data report
src/baseline_tfidf.py   TF-IDF baselines, 5-fold CV, leaky-vs-clean comparison
notebooks/              fine-tuning notebook (Kaggle/Colab GPU)  TODO
app/                    Gradio web app                            TODO
results/                metrics, confusion matrices, error files
reports/                final report                              TODO
```

## Methodology, experiments, results, error analysis, limitations
TODO

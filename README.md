# Kirundi News Topic Classification (KIRNEWS, AfroXLMR)

Individual NLP capstone (Summative Project Sep26): classify Kirundi news articles into 12 topics.

## Project links
| Resource | Link |
|---|---|
| GitHub repository | https://github.com/Delucie-Rurangwa/kirundi-news-classification |
| Live system (Hugging Face Space) | https://huggingface.co/spaces/delucie/kirundi-news-demo |
| Fine-tuned model | https://huggingface.co/delucie/kirundi-news-afroxlmr |
| Demo video | https://www.awesomescreenshot.com/video/57562867|

## Problem
Automatically assign a Kirundi news article to one of 12 topics: politics, sport, economy, health, entertainment, history, technology, culture, religion, environment, education, relationship. Kirundi is a low-resource Bantu language, mutually intelligible with Kinyarwanda.

## Dataset
KIRNEWS (Niyongabo et al., 2020, COLING): 4,612 articles from 8 Burundian news sources, 12 topics. Source and data: https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus.
The copyright of the articles belongs to the original news sources, so the data is **not** included in this repository. Download `KIRNEWS.zip` from the authors' page and unzip it into `data/raw/` so that `data/raw/KIRNEWS/raw/train.csv` exists.

**Data-quality findings** (from `src/data_prep.py`): only 1,811 of 4,612 rows are unique texts; all 589 unique official test articles also occur in the training file; 35 texts carry conflicting labels (one election article appears 172 times under 3 labels). After removing duplicates and conflicts, 1,776 articles remain and are split 70/15/15 (stratified, seed 42): 1,243 train / 266 validation / 267 test. The classes are strongly imbalanced (politics 611 ... history 9).

## Method
- **Baselines:** TF-IDF (word, char, word+char) with logistic regression / linear SVM, 5-fold cross-validation.
- **Transformers:** XLM-R-base vs AfroXLMR-base (`Davlan/afro-xlmr-base`), fine-tuned end to end with early stopping on validation macro-F1; optional class-weighted loss; 256 or 512 tokens; 3 seeds each.
- **Evaluation:** macro-F1 (main), accuracy, weighted-F1; paired bootstrap on the same test articles.

## Results (clean split)
| Config | Test macro-F1 (mean ± std, 3 seeds) | Test accuracy |
|---|---|---|
| E1: XLM-R-base, plain loss, 256 tokens | 0.371 ± 0.015 | 0.777 |
| E2: AfroXLMR-base, plain loss, 256 tokens | 0.701 ± 0.024 | 0.908 |
| **E3: AfroXLMR-base, class-weighted, 256 tokens** | **0.766 ± 0.007** | 0.921 |
| E4: AfroXLMR-base, class-weighted, 512 tokens | 0.740 ± 0.015 | 0.928 |

| TF-IDF baseline | Test macro-F1 | CV macro-F1 (5-fold) |
|---|---|---|
| Word LR | 0.615 | 0.718 ± 0.034 |
| Char SVM | 0.812 | 0.755 ± 0.047 |
| Word+char SVM | 0.763 | 0.752 ± 0.045 |

**Final model** (E3 config, seed chosen on validation): test accuracy 0.906, macro-F1 0.778. A paired bootstrap shows **no reliable difference** from the best TF-IDF baseline (difference in macro-F1 -0.035, 95% interval -0.159 to +0.049). See the report for the error analysis and limitations.

## Repository structure
```
src/data_prep.py            audit, de-duplicate, remap labels, stratified split, data report
src/baseline_tfidf.py       TF-IDF baselines, 5-fold CV, official-vs-clean comparison (--skip_cv for a fast run)
src/finetune_transformer.py Transformer fine-tuning (GPU): weighted loss, early stopping, result export
src/aggregate_results.py    experiments table (mean ± std over seeds)
src/compare_bootstrap.py    paired bootstrap comparison and per-class F1
notebooks/                  kaggle_finetune.ipynb, colab_finetune.ipynb
app/                        Gradio web app (app.py, requirements.txt, Space README)
results/                    metrics, confusion matrices, error files
requirements.txt            CPU requirements (data prep, baselines)
requirements-gpu.txt        GPU requirements (fine-tuning)
```

## Reproduce
**1. Data and baselines (CPU, laptop).**
```bash
pip install -r requirements.txt
python src/data_prep.py --version raw
python src/baseline_tfidf.py --version raw          # about 5-10 min with cross-validation
python src/baseline_tfidf.py --version raw --skip_cv   # seconds, saves test predictions
```
**2. Fine-tuning (GPU: Kaggle or Colab).** Upload the files from `data/processed/` as a *private* Kaggle dataset (the article copyright belongs to the news sources), open `notebooks/kaggle_finetune.ipynb`, and run the cells in order: smoke test, then 12 runs (4 configs x 3 seeds, about 1.5-2 hours on two T4 GPUs), then `src/aggregate_results.py`. Results appear under `results/transformers/`.

**3. Compare with the baseline.**
```bash
python src/compare_bootstrap.py --a results/preds_tfidf_char_svm_raw.csv --a_name "char SVM" \
  --b results/transformers/preds_E3_afroxlmr_base_len256_wloss_s44.csv --b_name "AfroXLMR E3"
```

## Deployment
1. Upload the final model folder to the Hugging Face Hub (`HfApi.upload_folder`) as `delucie/kirundi-news-afroxlmr`.
2. Create a Hugging Face Space (SDK: Gradio, free CPU) and upload `app/app.py`, `app/requirements.txt` and `app/README.md`.
3. In the Space settings add the variable `MODEL_ID = delucie/kirundi-news-afroxlmr`.
4. Run locally with `MODEL_ID=delucie/kirundi-news-afroxlmr python app/app.py`.

## Existing resources vs own work
Used: KIRNEWS data, XLM-R and AfroXLMR weights, scikit-learn, Transformers, PyTorch, Gradio, Kaggle GPUs, Hugging Face Hub. Implemented by me: data audit and cleaning, baselines, fine-tuning pipeline, experiments, bootstrap analysis, error analysis, web app. An AI assistant (Claude) helped debug code; I ran and checked all experiments (see the report's declaration).

## References
Niyongabo et al. (2020) KINNEWS and KIRNEWS, COLING. Conneau et al. (2020) XLM-R, ACL. Alabi et al. (2022) AfroXLMR, COLING. Berg-Kirkpatrick et al. (2012) significance testing in NLP, EMNLP. Full list in the report.

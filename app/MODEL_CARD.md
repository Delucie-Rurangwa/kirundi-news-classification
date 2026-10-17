---
language: rn
license: other
base_model: Davlan/afro-xlmr-base
tags: [text-classification, kirundi, african-languages, news]
---
# Kirundi news topic classifier (AfroXLMR-base fine-tuned on KIRNEWS)

- **Task:** 12-class news topic classification for Kirundi.
- **Base model:** Davlan/afro-xlmr-base (check its license on the Hub before redistribution).
- **Data:** KIRNEWS (Niyongabo et al., 2020, COLING). Article copyright belongs to the original news sources; this model is released for research/educational use. Training used a de-duplicated split (1,243 train / 266 val / 267 test articles).
- **Training:** class-weighted cross-entropy, lr 3e-5, 12 epochs max with early stopping, max length 256 tokens.
- **Limitations:** very small data; rare classes (history, religion, culture) are unreliable; not tested on other news domains or on Kinyarwanda.
- **Code:** https://github.com/Delucie-Rurangwa/kirundi-news-classification

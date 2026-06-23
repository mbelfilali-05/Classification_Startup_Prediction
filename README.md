# Startup Outcome Classification

Predicting the eventual fate of a startup — **acquired, IPO, or closed** — from its early funding history, sector, geography, and timing signals. A full, reproducible machine-learning study that benchmarks classical ML against modern deep-learning architectures on tabular data, with rigorous cross-validation, hyperparameter tuning, and error analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-success)
![PyTorch](https://img.shields.io/badge/PyTorch-TabNet%20%7C%20TabTransformer-red)
![Status](https://img.shields.io/badge/status-research%20complete-brightgreen)

> **Headline result:** A stacked gradient-boosting ensemble reaches **0.7515 macro-F1** on the held-out test set for the binary task and **0.6727 macro-F1** for the 3-class task — **outperforming every deep-learning architecture tried**, including TabNet, TabTransformer, FT-Transformer, and tuned MLPs.

---

## Table of contents

- [Motivation](#motivation)
- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key findings](#key-findings)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Reproducing the study](#reproducing-the-study)
- [Tech stack](#tech-stack)
- [Roadmap](#roadmap)
- [Authors](#authors)
- [License & acknowledgements](#license--acknowledgements)

---

## Motivation

Most of a startup's fate is decided early — by how fast it raises, how much, in what sector, and where. This project asks a concrete question: **using only signals available from a company's funding record, how well can we predict whether it will be acquired, go public, or shut down?**

Beyond the prediction itself, the project is a controlled experiment in *modeling strategy*: it treats "which model family wins on structured tabular data, and why" as the real research question, and answers it with a clean, leak-free experimental protocol rather than a single lucky run.

## Problem statement

The raw label is a startup's `status` ∈ `{operating, closed, acquired, ipo}`. Two complementary framings are studied:

| Framing | Classes | What it answers |
|---|---|---|
| **Binary** | success *(acquired / IPO)* vs. closed | Will this company reach a positive exit at all? |
| **3-class** | closed / acquired / IPO | What *kind* of outcome, distinguishing the two success modes? |

Both are evaluated with **macro-F1**, chosen deliberately: the classes are imbalanced (IPOs are rare), and macro-F1 refuses to let a model win by ignoring the minority class.

## Dataset

- **Primary source:** `big_startup_secsees_dataset` — 66,368 startup records, 14 raw columns (funding totals, rounds, founding/funding dates, sector tags, country/region/city, status).
- **Secondary source:** `CAX_Startup_Data` — used for cross-checking and enrichment.
- Records are filtered to those with a **resolved outcome** for the supervised task, then split **70 / 15 / 15** into train / validation / test with a fixed seed (42). The test set is touched **once**, at the very end.

## Methodology

The pipeline is built in the *"notebooks for looking, files for building"* style: exploration lives in notebooks, but anything trusted and reused is extracted into importable, tested modules. Every hyperparameter lives in a single `params.yaml` — no magic numbers scattered across cells.

**1 — Feature engineering (`feature_engineering.py`).** A Géron-style scikit-learn pipeline turns 14 raw columns into **21 model-ready features** (11 numeric, 7 binary, 3 categorical), all fit on training data only to prevent leakage:

- *Funding signals:* `log` funding total, funding per round, rounds², funding velocity (log).
- *Timing signals:* company age, time-to-first-funding, funding duration, funding recency.
- *Velocity:* rounds per year.
- *Sector & geography:* rare-category grouping into "Other", top-N countries/regions, biotech and top-IPO-region flags.

**2 — Baselines (`nb03a`).** Logistic Regression, SVM, Random Forest, Extra Trees, and the gradient-boosting trio (XGBoost, LightGBM, CatBoost) under identical cross-validation.

**3 — Class imbalance (`nb03b`).** SMOTE vs. class weights vs. no rebalancing, compared honestly on macro-F1 rather than accuracy.

**4 — Hyperparameter tuning (`nb03c`).** Systematic tuning of the strongest gradient-boosting models, plus a feature-ablation study to measure what each feature group is actually worth.

**5 — Ensembling (`nb03d`).** Soft-voting and stacking over the tuned boosters.

**6 — Deep learning (`nb04a`, `nb04b`).** A fair counter-experiment: MLPs (wide/deep), **TabNet**, **TabTransformer**, and **FT-Transformer**, tuned under the same budget and seed.

**7 — Final comparison (`nb05`).** ML vs. DL on the untouched test set, with confusion matrices and a dedicated **IPO-boundary analysis** of where the hard errors concentrate.

## Results

**Final held-out test set (macro-F1):**

| Framing | Family | Best model | Test macro-F1 |
|---|---|---|---|
| Binary | **ML** | Stacking (XGB + LGBM + RF → LR) | **0.7515** |
| Binary | DL | TabTransformer-Small | 0.7370 |
| 3-class | **ML** | XGBoost (tuned) | **0.6727** |
| 3-class | DL | TabTransformer-Small | 0.6332 |

**Cross-validation leaderboard (top of each framing):** binary peaks at XGBoost-tuned 0.7513 and Stacking 0.7501; 3-class peaks at LightGBM-tuned 0.6677 and XGBoost-tuned 0.6666. The best DL architecture (TabTransformer-Small) matches the *baseline* boosters but never overtakes the *tuned* ones.

The complete 55-row experiment log lives in [`master_scoreboard.csv`](master_scoreboard.csv); per-stage results are in the `ml_*.csv` / `dl_*.csv` files.

## Key findings

1. **Gradient-boosted trees beat deep learning on this data — consistently.** Across both framings and every architecture, tuned XGBoost/LightGBM and their ensembles win. This reproduces the broader literature finding that tree ensembles remain state-of-the-art on heterogeneous tabular data, and it held even after giving the DL models a fair tuning budget.

2. **Timing features carry the signal.** The feature-ablation study ([`feature_ablation.csv`](feature_ablation.csv)) shows that removing the **timing** group costs the most (**−0.039 macro-F1**), far more than sector (−0.016) or geography (−0.006). TabNet's learned feature importances agree: *funding recency* and *time-to-first-funding* are the top two attributions — **when** a company raises is more predictive than **how much**.

3. **The hard error is IPO vs. acquired.** The [`ipo_boundary_analysis.csv`](ipo_boundary_analysis.csv) shows the 3-class ceiling is set by confusion between the two *success* modes: the best model recovers IPOs at 0.57 recall but misroutes a third of them to "acquired" — the two classes look nearly identical in funding space. This, not the closed class, is what caps 3-class performance.

4. **Framing changes the problem more than the model does.** Collapsing to binary lifts macro-F1 by ~0.08 regardless of model — a reminder that *how you pose the question* is a first-class modeling decision, not an afterthought.

## Repository structure

```
startup-classification/
├── data/                       # raw + processed + splits (git-ignored)
├── notebooks/
│   ├── nb01_eda_raw.ipynb       # exploration of the raw data
│   ├── nb02_preprocessing.ipynb # cleaning + feature engineering
│   ├── nb03a_baselines.ipynb    # LogReg, RF, SVM, XGB/LGBM/CatBoost
│   ├── nb03b_imbalance.ipynb    # SMOTE vs class weights
│   ├── nb03c_tuning.ipynb       # tuning + feature ablation
│   ├── nb03d_ensemble.ipynb     # voting + stacking
│   ├── nb04a_dl_architectures.ipynb  # MLP, TabNet, TabTransformer, FT-Transformer
│   ├── nb04b_dl_tuning.ipynb    # DL tuning (GPU)
│   └── nb05_comparison.ipynb    # ML vs DL on the test set
├── feature_engineering.py       # the production feature pipeline
├── params.yaml                  # single source of truth for all hyperparameters
├── results/                     # scoreboards, ablations, test metrics (CSV)
├── reports/figures/             # confusion matrices, comparison plots (PNG)
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Clone
git clone https://github.com/<your-username>/startup-classification.git
cd startup-classification

# 2. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Place the raw dataset under data/raw/
#    big_startup_secsees_dataset.csv
```

## Reproducing the study

Run the notebooks in order — each consumes the previous stage's output and reads its settings from `params.yaml`:

```text
nb01_eda  →  nb02_preprocessing  →  nb03a_baselines  →  nb03b_imbalance
          →  nb03c_tuning  →  nb03d_ensemble  →  nb04a_dl_architectures
          →  nb04b_dl_tuning  →  nb05_comparison
```

The deep-learning notebooks (`nb04a/b`) are written to run on a free Colab/Kaggle GPU; everything else runs comfortably on CPU. All randomness is seeded (`random_seed: 42`) so the numbers above are reproducible.

## Tech stack

`Python` · `pandas` / `NumPy` · `scikit-learn` (pipelines, ColumnTransformer) · `XGBoost` · `LightGBM` · `CatBoost` · `imbalanced-learn` (SMOTE) · `PyTorch` · `pytorch-tabnet` · `TabTransformer` / `FT-Transformer` · `Matplotlib` / `Seaborn` · `PyYAML`

## Roadmap

The research phase is complete. Planned next steps turn the best model into a usable artifact:

- [ ] Serialize the winning pipeline (`pipeline.pkl` + tuned booster) and add a `predict.py` entry point.
- [ ] **FastAPI** service exposing a `/predict` endpoint.
- [ ] **Streamlit** demo so a non-technical user can score a startup interactively.
- [ ] **Dockerfile** for one-command reproducible deployment.
- [ ] Unit tests (`tests/`) covering the cleaning logic and the prediction path.
- [ ] SHAP-based explanations surfaced in the demo to make predictions interpretable.

## Authors

**Mohamed Belfilali Mimoun** · **Walid Fadli**
Data Science & Artificial Intelligence engineering students — *Projet de Fin d'Année (PFA) 2025–2026*, ENSAM-Rabat.

## License & acknowledgements

Released under the MIT License — see `LICENSE`.

Dataset: *Big Startup Success/Fail* (Crunchbase-derived) and the CAX startup dataset. The modeling methodology follows the bias-variance discipline and pipeline structure of Géron's *Hands-On Machine Learning*.

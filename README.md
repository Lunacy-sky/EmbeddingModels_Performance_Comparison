# LLM Answer Quality Prediction — Embeddings + ML

A study that automatically predicts the human-assigned quality score (**1–5**) of answers produced by a Turkish large language model (**CosmosLLM**). Texts are converted into vectors using different **embedding models**, and the score is then predicted with classical **machine learning regression** models. The project systematically compares which text input, which embedding model, and which ML algorithm yield the best results.

> Artificial Intelligence 
> Hasan Subaşı

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Method](#method)
- [Experimental Design](#experimental-design)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Sample Results](#sample-results)
- [Project Structure](#project-structure)

---

## Overview

This is a **regression** problem: given a question–reasoning–answer triple, predict the 1–5 score a human would assign to that answer. The textual labels (`çok kötü … çok iyi`, i.e. "very bad" … "very good") are mapped to numeric scores:

| Label (Turkish) | Meaning | Score |
|---|---|---|
| çok kötü | very bad | 1 |
| kötü | bad | 2 |
| orta | average | 3 |
| iyi | good | 4 |
| çok iyi | very good | 5 |

Pipeline: encode the text with an embedding model → train an ML regressor on those vectors → evaluate on the test set using RMSE, MAE, and R².

The dataset itself was collected from **real users** and stored in the `LLM_odev2_veriler.xlsx` file. Because computing the embeddings over this dataset is computationally heavy, the embeddings were generated on **Google Colab** using a helper script (leveraging Colab's GPU), and the resulting vectors were cached for the downstream ML stage.

## Dataset

The data lives in `LLM_odev2_veriler.xlsx` as a form-responses table (~**12,000** rows). Columns used:

| Column | Description |
|---|---|
| `Sorunuz` | The question the user asked the LLM |
| `CosmosLLM düşünme süreci` | The model's reasoning / thinking process |
| `CosmosLLM cevabı` | The model's final answer |
| `Değerlendirme Puanınız` | Human evaluation label (target variable) |

Preprocessing: empty texts are filled with empty strings, score labels are mapped to 1–5, and rows with unmatched/missing scores are dropped. The data is split randomly into **1,000** test rows and the rest for training (`RANDOM_SEED=42`).

## Method

The study tries all combinations of three factors.

**Text configurations (5):** which text is fed to the model.

| Config | Input |
|---|---|
| `S` | Question |
| `D` | Reasoning process |
| `S+D` | Question + Reasoning |
| `D+C` | Reasoning + Answer |
| `S+C` | Question + Answer |

**Embedding models (4):** loaded via `sentence-transformers`.

| Name | Model |
|---|---|
| Turkish-E5-Large | `ytu-ce-cosmos/turkish-e5-large` |
| Jina-v5-Small | `jinaai/jina-embeddings-v5-text-small` |
| Harrier-0.6B | `microsoft/harrier-oss-v1-0.6b` |
| Qwen3-Embed-0.6B | `Qwen/Qwen3-Embedding-0.6B` |

**ML regression models (3):**

| Model | Settings |
|---|---|
| Ridge | `StandardScaler` + L2 (alpha=1.0) |
| RandomForest | 200 trees |
| SVR | `StandardScaler` + RBF kernel |

Embedding outputs are cached as `.pkl` files in `embed_cache/` so they are not recomputed on repeated runs. A GPU (`cuda`) is used automatically when available. Given the size of the real-user dataset, the embedding step was run on **Google Colab** (with GPU) through a helper script, and the cached vectors were reused locally for the ML training and evaluation stages.

## Experimental Design

A total of **5 × 4 × 3 = 60 experiments** are run. For each experiment, three metrics are computed on both the training and test sets:

- **RMSE** (Root Mean Squared Error) — lower is better
- **MAE** (Mean Absolute Error) — lower is better
- **R²** (coefficient of determination) — higher is better

## Installation

Python 3.10+ recommended.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
pip install sentence-transformers torch
```

> Some embedding models require `trust_remote_code`; models are downloaded on first run. A GPU-enabled environment (e.g. CUDA) significantly speeds up embedding computation.

## Usage

`LLM_odev2_veriler.xlsx` must be in the same folder as the script.

```bash
python 23011073_YZ_Odev2.py
```

The script loads the data, performs the train/test split, runs the 60 experiments (caching embeddings), prints summary tables to the terminal, and saves the figures and result CSV into the `results/` folder.

## Outputs

Generated in the `results/` folder:

| File | Content |
|---|---|
| `tum_sonuclar.csv` | All metrics for the 60 experiments |
| `fig1_heatmap_config_embed.png` | Config × Embedding heatmaps (RMSE/MAE/R²) |
| `fig2_heatmap_ml_embed.png` | ML algorithm × Embedding heatmaps |
| `fig3_bar_factor_effects.png` | Effect of each factor on test performance (bar charts) |
| `fig4_all60_experiments.png` | All 60 experiments in one chart (Test RMSE) |
| `fig5_boxplots.png` | Test RMSE distributions per factor |
| `fig6_train_vs_test.png` | Train vs Test RMSE — overfitting analysis |
| `fig7_top_bottom10.png` | Best and worst 10 combinations |

## Sample Results

The combinations achieving the lowest Test RMSE in one run (`results/tum_sonuclar.csv`):

| Config | Embedding | ML | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|---|
| S+C | Harrier-0.6B | RandomForest | 1.0639 | 0.8283 | 0.1300 |
| S+C | Turkish-E5-Large | RandomForest | 1.0718 | 0.8265 | 0.1169 |
| S+C | Qwen3-Embed-0.6B | RandomForest | 1.0883 | 0.8406 | 0.0895 |

> Results may vary depending on environment and model versions; the values above are taken from the sample CSV included in the repo.

## A Note on the Embedding Cache

Due to GitHub's file-size limits, **no embedding cache files are included** in this repository — neither training nor test (`embed_cache/` is empty in the repo). The cached vectors are large derived artifacts and are not version-controlled.

When you run the script, the embeddings are computed and written to `embed_cache/` automatically (or can be precomputed on Google Colab as described above). In other words, the cache is regenerated from the source data and the embedding models; it does not need to ship with the repository.

## Project Structure

```
.
├── Comparison_Embedding_Models.py          # Main source code
├── LLM_odev2_veriler.xlsx        # Dataset (form responses)
├── results/                      # Generated figures and tum_sonuclar.csv
├── embed_cache/                  # Embedding cache (.pkl) — not in repo; regenerated on run
├── .vscode/
│   └── settings.json
└── README.md
```

<div align="center">

# 📈 Semiconductor News Sentiment & Stock Movement

### *Can NLP models predict stock direction from news headlines?*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![VADER](https://img.shields.io/badge/VADER-Sentiment-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![NTU](https://img.shields.io/badge/NTU-Advanced%20NLP-blueviolet?style=for-the-badge)

<br>

*A deep learning project exploring how positive and negative sentiment in semiconductor industry news affects stock price movements, using BiLSTM and Transformer architectures.*

---

</div>

## 🎯 Project Overview

Financial markets are highly sensitive to news. This project investigates whether **NLP-based sentiment analysis** on semiconductor industry news can predict short-term stock price direction (up vs. down).

We collected news headlines covering major semiconductor companies (NVIDIA, AMD, Intel, TSMC, Broadcom, and more), labeled them for sentiment using **VADER**, and trained deep learning models to classify sentiment as **positive vs. non-positive**. We then linked sentiment signals to actual stock returns over 1-day, 3-day, and 5-day horizons.

## 🏗️ Pipeline

```
📰 News Headlines ──→ 🧹 Preprocessing ──→ 🏷️ VADER Labeling ──→ 🧠 Deep Learning Models ──→ 📊 Stock Signal
                                              (pos / non-pos)       (BiLSTM & Transformer)
```

## 📂 Dataset

| Detail | Info |
|---|---|
| **Source** | Semiconductor industry news (Nov–Dec 2025) |
| **Tickers Covered** | NVDA, AMD, INTC, TSM, AVGO, QCOM, MU, TXN, ASML, AMAT, LRCX, KLAC, ADI, NXPI, MRVL, ON, MCHP, TER |
| **Benchmarks** | SOXX, SMH |
| **Labeling** | VADER sentiment (positive = 1, non-positive = 0) |
| **Total Samples** | 292 labeled news articles |
| **Split** | Train: 123 · Val: 71 · Test: 98 (time-based split) |

## 🧠 Models

### 1️⃣ BiLSTM (Bidirectional LSTM)

A bidirectional LSTM that reads news text forwards and backwards to capture contextual sentiment patterns.

**Architecture:** Embedding → Bidirectional LSTM → Dropout → FC → Sigmoid

| Hyperparameter | Best Config |
|---|---|
| Embedding Dim | 256 |
| Hidden Dim | 256 |
| Num Layers | 2 |
| Dropout | 0.2 |
| Learning Rate | 5e-4 |
| Batch Size | 32 |
| Pos Weight | No |

**Best Results (Grid Search — 720 configs):**

| Metric | Validation | Test |
|---|---|---|
| **Accuracy** | 60.6% | 58.2% |
| **F1 Score** | 0.650 | 0.661 |
| **ROC-AUC** | 0.689 | 0.628 |
| **Precision** | 0.531 | 0.548 |
| **Recall** | 83.9% | 83.3% |

---

### 2️⃣ Transformer Encoder

A custom Transformer encoder with positional encoding and mean-pooling over non-padded tokens for classification.

**Architecture:** Embedding → Positional Encoding → Transformer Encoder → Mean Pool → Dropout → FC → Sigmoid

| Hyperparameter | Best Config |
|---|---|
| d_model | 64 |
| Attention Heads | 2 |
| Feed-Forward Dim | 512 |
| Num Layers | 1 |
| Dropout | 0.4 |
| Learning Rate | 1e-3 |
| Batch Size | 16 |
| Pos Weight | Yes |

**Best Results (Grid Search — 3,888 configs):**

| Metric | Validation | Test |
|---|---|---|
| **Accuracy** | 80.3% | — |
| **F1 Score** | 0.788 | — |
| **ROC-AUC** | 0.806 | — |
| **Precision** | 0.743 | — |
| **Recall** | 83.9% | — |

> 💡 **Threshold tuning** was applied — the optimal decision threshold was selected on validation data by maximizing F1 (threshold = 0.375 instead of default 0.5), which significantly improved recall for both models.

## 📁 Repository Structure

```
📦 Natural-Language-Processing-NLP-
├── 📓 bilstm.ipynb                  # BiLSTM training, grid search & evaluation
├── 📓 transformer.ipynb             # Transformer training, grid search & evaluation
├── 📄 Group 22-Final Report.pdf     # Full project report
├── 📂 Data/
│   ├── labelled_semiconductor_news_vader.csv
│   ├── semiconductor_prices_oct2025_long.csv
│   ├── semiconductor_prices_oct2025_with_returns.csv
│   ├── news_price_direction_dataset.csv
│   ├── bilstm_grid_results.csv
│   ├── bilstm_test_predictions.csv
│   ├── transformer_grid_results.csv
│   ├── transformer_test_predictions.csv
│   └── transformer_best_config.json
└── 📂 Models/
    ├── bilstm_sentiment.pt
    ├── bilstm_best.pt
    ├── bilstm_price_direction_best.pt
    ├── transformer_sentiment.pt
    └── transformer_best.pt
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn yfinance
```

### Run the Notebooks

1. **BiLSTM** — Open `bilstm.ipynb` and run all cells
2. **Transformer** — Open `transformer.ipynb` and run all cells

> Both notebooks include data loading, preprocessing, model training, hyperparameter grid search, and evaluation.

## 🔑 Key Takeaways

- The **Transformer encoder outperformed BiLSTM** on validation metrics (ROC-AUC: 0.806 vs 0.689)
- **Threshold tuning** was critical — the default 0.5 cutoff missed many positive-sentiment articles
- **VADER labeling** provided a reasonable but noisy ground truth, which limited ceiling performance
- Small dataset size (292 samples) was a key challenge — data augmentation or pretrained embeddings could help future work

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-Stock%20Data-blue?style=flat-square)

</div>

## 👥 Team — Group 22

| Name |
|---|
| Joshua Goh |
| Akiko Orui |
| He YangYang |
| LongYu |
| Alluri Kanish |
| Mohamed AlNuwais |

<div align="center">

*Built for **Advanced NLP with Deep Learning** — Nanyang Technological University (NTU), 2025*

</div>

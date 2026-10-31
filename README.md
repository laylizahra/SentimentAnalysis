# 🎬 Sentiment Analysis of Movie Reviews Using Chunking & RoBERTa-Large

> Final Project of **Introduction to Data Science 2026**
> Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia

This project classifies English movie reviews as **positive** or **negative**.
We fine-tune the **RoBERTa-large** model and use a **chunking** method so that long reviews can be read completely by the model.

---

## 📑 Table of Contents

1. [Team Members](#-team-members)
2. [Project Overview](#-project-overview)
3. [Results](#-results)
4. [Repository Structure](#️-repository-structure)
5. [Dataset](#-dataset)
6. [Method / Pipeline](#️-method--pipeline)
7. [Why RoBERTa-Large?](#-why-roberta-large)
8. [Hyperparameters](#-hyperparameters)
9. [How to Run](#-how-to-run)
10. [Requirements](#-requirements)
11. [Citation](#-citation)
12. [References](#-references)

---

## 👥 Team Members

| No | Name | Email |
|---|---|---|
| 1 | Layli Nurul Fatimah Zahra | laylizahra28@gmail.com |
| 2 | Raychana Maharani Zahra | zahraychana@gmail.com |
| 3 | Femita Pamudyawati | femipamudji05@gmail.com |
| 4 | Chyntia Dwi Rahma | chyntiadwi83@gmail.com |
| 5 | Bunga Berliant Aulia Jauhari | bungabaj2005@gmail.com |
| 6 | Aida Belva Fitria | aidabelva04@gmail.com |

---

## 📌 Project Overview

### The Problem
Movie reviews are often very long. Transformer models like RoBERTa can only read **512 tokens** at one time.
If we simply cut the text, the model loses part of the review — and sometimes the most important part (for example, the conclusion at the end).

### Our Solution
We use a **sliding-window chunking** method:

1. Split each review into several **overlapping chunks** (512 tokens each, 256 tokens overlap).
2. The model predicts the sentiment of **each chunk**.
3. Combine all chunk predictions into **one final prediction** using **confidence-weighted pooling** (chunks with more confident predictions get more weight).

With this method, **no part of the review is thrown away**.

---

## 📊 Results

The model is evaluated with **5-Fold Stratified Cross-Validation**. The scores below are **Out-of-Fold (OOF)** scores, which means every training sample is predicted by a model that did not see it during training.

| Metric | Score |
|---|---|
| **OOF Accuracy** | **98.2%** |
| OOF Macro F1 | 0.9820 |
| OOF Precision (macro) | 0.9820 |
| OOF Recall (macro) | 0.9821 |

---

## 🗂️ Repository Structure

```
SentimentAnalysis/
├── roberta_sentiment.ipynb            # Main notebook (ready to run on Kaggle)
├── Report of Sentiment Analysis.pdf   # Full project report (paper)
└── README.md                          # Project documentation (this file)
```

> 💡 The notebook creates a file called `submission.csv` (columns: `Row`, `Label`) after it finishes running.

---

## 📁 Dataset

The data comes from the [Kaggle competition: Final Project Introduction to Data Science 2026](https://www.kaggle.com/competitions/final-project-introduction-to-data-science-2026).

| Split | Samples | Positive | Negative |
|---|---|---|---|
| Train | 1,500 | 752 (50.1%) | 748 (49.9%) |
| Test | 500 | — | — |

- **Input:** `text` — the movie review (English)
- **Target:** `label` — `positive` or `negative`

The classes are almost perfectly balanced, so **accuracy** is a fair main metric.
We still use small class weights in the loss function for safety.

---

## ⚙️ Method / Pipeline

```
 Raw review text
       │
       ▼
 ┌──────────────────────┐
 │ 1. Tokenization      │  RoBERTa tokenizer (fast)
 └──────────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ 2. Chunking          │  512 tokens per chunk, stride 256, max 8 chunks
 └──────────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ 3. RoBERTa-Large     │  Predicts probability for each chunk
 └──────────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ 4. Chunk Pooling     │  Confidence-weighted average → 1 score per review
 └──────────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ 5. Threshold Tuning  │  Best threshold searched from 0.30 to 0.70
 └──────────────────────┘
       │
       ▼
 ┌──────────────────────┐
 │ 6. 5-Fold Ensemble   │  Average test predictions from 5 models
 └──────────────────────┘
       │
       ▼
   submission.csv
```

### Step-by-step explanation

**1. Tokenization**
The text is converted into tokens with the `roberta-large` tokenizer.

**2. Chunking**
Long reviews are split into chunks of 512 tokens. Each new chunk starts 256 tokens after the previous one, so neighbouring chunks share context. We keep up to 8 chunks per review.

**3. Fine-tuning RoBERTa-Large**
Training uses several techniques to make learning stable:
- **AdamW optimizer** with weight decay
- **Layer-wise learning rate decay (LLRD):** lower layers learn more slowly than upper layers, so general language knowledge is kept
- **Cosine learning rate schedule** with warmup
- **Gradient accumulation** (effective batch size = 32)
- **Mixed precision (AMP)** to save GPU memory and time
- **Gradient clipping** (max norm = 1.0)
- **Label smoothing** and **class-weighted cross-entropy loss**
- The best model of each fold is saved based on the **lowest validation loss**

**4. Chunk Pooling**
Each chunk gives a probability. The final review probability is a weighted average, where the weight is the confidence of each chunk:

```
weight_i  = max(p_i)
P(review) = Σ (weight_i × p_i) / Σ weight_i
```

Other options in the code: `mean` and `max`.

**5. Threshold Tuning**
Instead of always using 0.5, we search for the threshold that gives the best accuracy on the OOF predictions.

**6. K-Fold Ensemble**
We train 5 models (one per fold). The final test prediction is the **average** of the 5 models' probabilities.

---

## 🧠 Why RoBERTa-Large?

RoBERTa (Liu et al., 2019) is an improved version of BERT:

- **More training data** — 160 GB of text (BERT uses 16 GB)
- **Dynamic masking** — different words are masked in every epoch
- **No Next Sentence Prediction** — this task was shown to be not helpful
- **Bigger batch sizes** — more stable training

The **large** version has **24 layers** and **355M parameters**. It understands context better than the base version, but it needs more GPU memory and time.

---

## 🔧 Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `MODEL_NAME` | `roberta-large` | Pre-trained model used |
| `MAX_LENGTH` | 512 | Max tokens per chunk |
| `STRIDE` | 256 | Overlap between chunks |
| `MAX_CHUNKS` | 8 | Max chunks per review |
| `BATCH_SIZE` | 4 | Batch size per GPU |
| `ACCUM_STEPS` | 8 | Gradient accumulation → effective batch 32 |
| `EPOCHS` | 4 | Epochs per fold |
| `N_SPLITS` | 5 | Number of Stratified K-Fold splits |
| `LR` | 7e-6 | Base learning rate |
| `LR_LAYER_DECAY` | 0.90 | Layer-wise LR decay factor |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `WARMUP_RATIO` | 0.10 | Warmup part of the cosine schedule |
| `PATIENCE` | 3 | Early stopping patience |
| `LABEL_SMOOTHING` | 0.05 | Label smoothing value (ε) |
| `POOLING_MODE` | `weighted` | How chunk predictions are combined |

---

## 🚀 How to Run

### Option A — Kaggle (recommended)
1. Open the [competition page](https://www.kaggle.com/competitions/final-project-introduction-to-data-science-2026) and create a new notebook.
2. Upload `roberta_sentiment.ipynb` (File → Import Notebook).
3. Turn on a **GPU** accelerator (Settings → Accelerator → GPU).
4. Click **Run All**.
5. After it finishes, download `submission.csv` from the output folder.

### Option B — Local machine
1. Clone this repository:
   ```bash
   git clone https://github.com/laylizahra/SentimentAnalysis.git
   cd SentimentAnalysis
   ```
2. Install the libraries (see [Requirements](#-requirements)).
3. Download `train.csv` and `test.csv` from Kaggle.
4. In the notebook, change `TRAIN_PATH` and `TEST_PATH` to your local file paths.
5. Run the notebook with Jupyter.

> ⚠️ RoBERTa-large is a big model. A GPU with **at least 16 GB VRAM** is strongly recommended.

---

## 📦 Requirements

- Python 3.9+
- PyTorch (with CUDA)
- Hugging Face Transformers
- scikit-learn
- NumPy, pandas
- Matplotlib

Install with:

```bash
pip install torch transformers scikit-learn numpy pandas matplotlib
```

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{2026sentimentroberta,
  title       = {A Sentiment Analysis of Movie Review Using RoBERTa-Large},
  author      = {Zahra, Layli Nurul Fatimah and Zahra, Raychana Maharani
                 and Pamudyawati, Femita and Jauhari, Bunga Berliant Aulia
                 and Rahma, Chyntia Dwi and Fitria, Aida Belva},
  year        = {2026},
  institution = {Institut Teknologi Sepuluh Nopember}
}
```

---

## 📚 References

1. M. Bordoloi and S. K. Biswas, "Sentiment Analysis: A Survey on Design Framework, Applications and Future Scopes," *Artificial Intelligence Review*, vol. 56, 2023.
2. Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.
3. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, 2019.
4. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019.
5. H. Bashiri and H. Naderi, "Comprehensive Review and Comparative Analysis of Transformer Models in Sentiment Analysis," *Knowledge and Information Systems*, vol. 66, 2024.

---

<p align="center">Made with ❤️ by the ITS Data Science 2026 Team</p>

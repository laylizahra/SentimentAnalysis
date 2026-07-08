# A Sentiment Analysis of Movie Review Using Chunking & RoBERTa-Large 

Final Project of Introduction to Data Science 2026 
Institut Teknologi Sepuluh Nopember (ITS), Surabaya.

Authors:
1. Layli Nurul Fatimah Zahra (laylizahra28@gmail.com)
2. Raychana Maharani Zahra (zahraychana@gmail.com)
3. Femita Pamudyawati (femipamudji05@gmail.com)
4. Chyntia Dwi Rahma (chyntiadwi83@gmail.com)
5. Bunga Berliant Aulia Jauhari (bungabaj2005@gmail.com)
6. Aida Belva Fitria (aidabelva04@gmail.com)

-----
# Overview
This repository contains the full pipeline for binary sentiment classification (positive / negative) of long-form English movie reviews using a fine-tuned RoBERTa-large transformer model.

The core challenge is that movie reviews routinely exceed the 512-token input limit of transformer architectures. This pipeline addresses that constraint through a sliding-window chunking strategy that partitions each review into overlapping token segments, processes each independently, and aggregates predictions via confidence-weighted probability pooling to ensuring no content is discarded.

Trained and evaluated on the [Kaggle competition dataset](https://www.kaggle.com/competitions/final-project-introduction-to-data-science-2026), the pipeline achieves:

| Metric | Score |
|---|---|
| **OOF Accuracy** | **98.2%** |
| OOF Macro F1 | 0.9820 |
| OOF Precision (macro) | 0.9820 |
| OOF Recall (macro) | 0.9821 |

---
## 🗂️ Repository Structure

```
├── roberta_sentiment.ipynb                 # Main notebook (Kaggle-ready)
├── submission.csv                          # Final predictions (Row, Label)
├── Report of Sentiment Analysis.pdf        # Report Paper 
└── README.md
```
---
## 🔧 Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `MODEL_NAME` | `roberta-large` | Backbone transformer |
| `MAX_LENGTH` | 512 | Max tokens per chunk |
| `STRIDE` | 256 | Overlap between chunks |
| `MAX_CHUNKS` | 8 | Max chunks per review |
| `BATCH_SIZE` | 4 | Per-GPU batch size |
| `ACCUM_STEPS` | 8 | Gradient accumulation → eff. batch 32 |
| `EPOCHS` | 4 | Max epochs per fold |
| `N_SPLITS` | 5 | Stratified K-Fold |
| `LR` | 7e-6 | Base learning rate |
| `LR_LAYER_DECAY` | 0.90 | Layer-wise LR decay factor |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `WARMUP_RATIO` | 0.10 | Cosine warmup fraction |
| `PATIENCE` | 3 | Early stopping patience |
| `LABEL_SMOOTHING` | 0.05 | Label smoothing ε |
| `POOLING_MODE` | `weighted` | Chunk aggregation strategy |

---
## 🧠 Why RoBERTa-Large?

RoBERTa (Liu et al., 2019) improves upon BERT through:
- **Larger training corpus** — 160GB of text vs BERT's 16GB
- **Dynamic masking** — different tokens masked each epoch
- **No next-sentence prediction** — removes a pretraining objective shown to be unhelpful
- **Larger batch sizes** — more stable gradient estimates

The `-large` variant (24 layers, 355M parameters) provides substantially richer contextual representations than `-base`, at the cost of higher memory and compute requirements.

---

## 📁 Dataset

| Split | Samples | Positive | Negative |
|---|---|---|---|
| Train | 1,500 | 752 (50.1%) | 748 (49.9%) |
| Test | 500 | — | — |

The near-perfectly balanced class distribution eliminates the need for aggressive class reweighting and allows the model to be evaluated with standard accuracy as the primary metric.

---
## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{2026sentimentroberta,
  title     = {Sentiment Analysis of Movie Review Using RoBERTa-Large 
               with Long-Text Chunking and Stratified Ensemble Learning},
  author    = {Zahra, Layli Nurul Fatimah and Zahra, Raychana Maharani 
               and Pamudyawati, Femita and Juhri, Bunga Berliant Aulia 
               and Rahma, Chyntia Dwi and Fitria, Aida Belva },
  year      = {2026},
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

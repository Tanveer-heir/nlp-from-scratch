# Knowledge Distillation — DistilBERT-Style

> Distill a small model from a larger fine-tuned one, with full speed/accuracy tradeoff analysis.

---

## Results

| Model | Accuracy | Acc% of Teacher | Speed (ms/sample) | Speedup | Params | Compression |
|---|---|---|---|---|---|---|
| BERT Teacher | 0.8925 | 100.0% | 0.87 | 1.00x | 109.5M | — |
| DistilBERT Baseline | 0.8750 | 98.0% | 0.34 | 2.58x | 67.0M | 38.8% |
| **DistilBERT Distilled** | **0.8850** | **99.2%** | **0.34** | **2.53x** | **67.0M** | **38.8%** |

**Key takeaway:** The distilled student recovers +1.00% accuracy over the naive baseline for free — same model size, same inference speed, just better training signal from the teacher's soft labels.

---

## Core Concept

Knowledge Distillation transfers the "dark knowledge" of a large **teacher** model into a compact **student** model. Rather than training the student on hard one-hot labels, it learns from the teacher's full probability distribution — which encodes rich inter-class relationships.

```
Hard label:   [0,    0,    1,    0,    0   ]   ← one-hot, discards information
Soft label:   [0.01, 0.02, 0.85, 0.08, 0.04]  ← teacher output, carries dark knowledge
```

### Loss Function

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y,\ \hat{y}_S) \;+\; (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{KL}\!\left(\sigma\!\left(\tfrac{z_T}{T}\right),\ \sigma\!\left(\tfrac{z_S}{T}\right)\right)$$

| Term | Role |
|---|---|
| $\alpha$ | Weight balancing hard-label vs soft-label loss |
| $T$ | Temperature — higher softens the distributions, revealing more dark knowledge |
| $T^2$ | Compensates for reduced gradient magnitude from temperature scaling |
| $\mathcal{L}_{CE}$ | Cross-entropy with ground truth |
| $\mathcal{L}_{KL}$ | KL-divergence between teacher and student soft distributions |

### Architecture: BERT → DistilBERT

```
BERT (Teacher)          DistilBERT (Student)
──────────────          ────────────────────
12 layers       →       6 layers       (50% fewer)
109.5M params   →       67.0M params   (38.8% smaller)
0.87 ms/sample  →       0.34 ms/sample (2.5x faster)
89.25% accuracy →       88.50% accuracy (99.2% retained)
```

---

## Project Structure

```
13_knowledge_distillation.ipynb   ← main notebook
README.md                         ← this file
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn numpy tqdm
```

### 2. Run the notebook

```bash
jupyter notebook 13_knowledge_distillation.ipynb
```

> **Dataset note:** The notebook uses SST-2 (Stanford Sentiment Treebank, binary sentiment classification). If you hit a `HfUriError` on `load_dataset("glue", "sst2")`, replace it with:
> ```python
> raw_dataset = load_dataset("stanfordnlp/sst2")
> ```

---

## Notebook Sections

| # | Section | What you learn |
|---|---|---|
| 1 | Data Loading | SST-2, custom `Dataset`, `DataLoader` |
| 2 | Teacher Fine-Tuning | Fine-tune BERT with AdamW + LR warmup |
| 3 | Student Baseline | DistilBERT trained with standard CE (no distillation) |
| 4 | Distillation Training | `DistillationLoss`, frozen teacher, combined hard+soft loss |
| 5 | Evaluation | Accuracy, inference speed, model size, compression ratio |
| 6 | Visualizations | Training curves, bubble chart, confusion matrices |
| 7 | Ablation Studies | Temperature sweep T∈{1,2,4,6,10}, alpha sweep α∈{0,0.2,0.5,0.7,1.0} |
| 8 | Advanced KD | Hidden-state alignment with cosine embedding loss (DistilBERT-style) |
| 9 | Variants Reference | All KD variants and when to use each |

---

## Key Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `temperature` | 4.0 | Controls softness of distributions. Higher → more dark knowledge revealed |
| `alpha` | 0.5 | Weight on hard-label CE loss. `1-alpha` weights the soft KL loss |
| `learning_rate` | 2e-5 | AdamW learning rate for both teacher and student |
| `distill_epochs` | 5 | Training epochs for the distilled student |
| `max_length` | 128 | Token sequence length |
| `batch_size` | 16 | Samples per batch |

### Temperature intuition

```
T=1  → [0.01, 0.00, 0.98, 0.01]   almost one-hot, little extra signal
T=4  → [0.14, 0.09, 0.55, 0.12]   spread out, inter-class info visible
T=10 → [0.20, 0.17, 0.30, 0.18]   nearly uniform, too much softening
```

### Alpha intuition

```
α=0.0  → pure soft labels (ignores ground truth entirely)
α=0.5  → balanced mix  ← default, works well in most cases
α=1.0  → pure hard labels (ignores teacher signal, equivalent to no distillation)
```

---

## Distillation Variants Covered

| Variant | Loss signal | Best for |
|---|---|---|
| **Response-based** | Teacher output logits (KL div) | Classification, any architecture |
| **Feature-based** | Intermediate hidden states (cosine/MSE) | When architectures are compatible |
| **Relation-based** | Pairwise sample similarities | Metric learning, representation learning |
| **Attention transfer** | Attention weight patterns | BERT-family, transformers |
| **Online / mutual** | Ensemble of peer students | No pre-trained teacher available |

---

## When to Use Knowledge Distillation

Edge or mobile deployment (memory/latency constraints)  
Real-time inference APIs  
Labeled data is scarce (teacher provides richer supervision than hard labels)  
Cost-sensitive production (smaller model = cheaper per-call)  
You have a strong teacher but can't serve it at scale  

### Complementary techniques

| Technique | Typical gain | Combines with KD? |
|---|---|---|
| Quantization (FP32→INT8) | ~4x speedup | Yes — stack for ~8x total |
| Pruning | 30–90% params removed | Yes |
| Architecture search (NAS) | Task-specific optimal size | Yes |
| Early exit | Dynamic per-sample speedup | Yes |

---

## References

- Hinton, Vinyals & Dean (2015) — [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Sanh et al. (2019) — [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- Jiao et al. (2019) — [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
- Romero et al. (2015) — [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)

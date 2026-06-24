# Word2Vec from Scratch — Skip-Gram + Negative Sampling

> **No PyTorch. No TensorFlow. No Gensim. Pure NumPy.**

A complete ground-up implementation of Word2Vec using the **Skip-Gram architecture with Negative Sampling (SGNS)**. Every component — the embedding matrices, loss function, gradient computation, and parameter updates — is written by hand so you can see exactly what's happening at each step.

---

## What It Learns

The model is never told what "royalty" or "capital city" means. It only sees raw co-occurrence patterns. Yet after training, the geometry of the embedding space recovers:

```
Nearest neighbors:   king  → queen, prince, princess
                     dog   → cat, puppy, kitten
                     paris → berlin, london, rome

Vector arithmetic:   king  − man  + woman  ≈  queen
                     paris − france + germany  ≈  berlin
                     prince − king + queen  ≈  princess
```

---

## Project Structure

```
word2vec_scratch.ipynb   ← Main notebook (run this)
README.md                ← This file
```

### Notebook sections

| # | Section | What happens |
|---|---------|-------------|
| 0 | Imports | NumPy, matplotlib, sklearn.PCA |
| 1 | Corpus & Tokenization | Raw text → token list |
| 2 | Vocabulary | Token counts, word↔index maps, unigram frequencies |
| 3 | Subsampling | Discard frequent words stochastically each epoch |
| 4 | Skip-gram Pairs | Sliding dynamic window → (center, context) pairs |
| 5 | Negative Sampling Table | Smoothed unigram^0.75 distribution, 1M-slot lookup |
| 6 | Model — Forward + Backward | Two embedding matrices, manual gradients |
| 7 | Training Loop | SGD + linear LR decay |
| 8 | Loss Curve | Convergence plot |
| 9 | Evaluation | Nearest neighbors, cosine similarities, analogies |
| 10 | PCA Visualisation | 2-D projection with colour-coded semantic clusters |

---

## Architecture

### Skip-Gram

Given a **center word**, predict its **surrounding context words**.

```
Center word:   "king"
Context pairs: (king, the), (king, rules), (king, queen), (king, with)
```

Skip-Gram is preferred over CBOW for rare words because each rare word
gets used as a center word and drives its own gradient updates.

### Two Embedding Matrices

| Matrix | Shape | Role |
|--------|-------|------|
| `W_in`  | V × D | Center word embeddings — **kept after training** |
| `W_out` | V × D | Context word embeddings — discarded after training |

Two matrices prevent the degenerate solution where a word is trivially
its own best context.

---

## Mathematics

### Negative Sampling Objective

Full softmax over the vocabulary costs O(V) per step. Negative Sampling
converts this into binary classification:

| Pair | Label |
|------|-------|
| (king, queen) — real co-occurrence | 1 |
| (king, cat) — random noise word | 0 |
| (king, ocean) — random noise word | 0 |

Each update costs O(k) regardless of vocabulary size, where k = 5–20.

### Loss Function

For center word **w**, positive context **c**, and k negatives **c⁻**:

```
L = −log σ(vw · vc)  −  Σᵢ log σ(−vw · vc⁻ᵢ)
```

The first term pulls the positive pair closer together.  
The second term pushes unrelated words apart.

### Manual Gradients

```
∂L/∂vw   = (σ(vw·vc) − 1)·vc  +  Σᵢ σ(vw·vc⁻ᵢ)·vc⁻ᵢ

∂L/∂vc   = (σ(vw·vc) − 1)·vw

∂L/∂vc⁻ᵢ = σ(vw·vc⁻ᵢ)·vw
```

No autograd. These are derived analytically from sigmoid BCE and
applied with plain SGD.

---

## Key Techniques

### Frequent-Word Subsampling

Words like "the", "is", "a" co-occur with everything and add noise.
Each token is discarded each epoch with probability:

```
P(discard w) = 1 − √(t / freq(w))     t ≈ 1e-3
```

High-frequency words are kept ~20% of the time; rare words almost always.
Re-running this each epoch acts as a form of data augmentation.

### Dynamic Context Window

The actual window size is sampled uniformly from `[1, window]` for each
center word. This implicitly gives closer words higher weight in the
aggregate training signal.

### Negative Sampling Distribution

Negatives are drawn from a smoothed unigram distribution:

```
P(w) ∝ freq(w)^0.75
```

| Power | Effect |
|-------|--------|
| 1.0 | Too peaked — common words dominate |
| 0.0 | Flat — ignores frequency entirely |
| **0.75** | Smooth compromise ✓ |

Implemented as a 1 000 000-slot lookup table for O(1) sampling.

### Learning Rate Schedule

Linear decay from `lr_start` to `lr_min` over all training steps,
matching the original Word2Vec C implementation:

```
lr(step) = max(lr_min,  lr_start × (1 − step / total_steps))
```

---

## Evaluation

### Cosine Similarity

```
cos(v₁, v₂) = (v₁ · v₂) / (‖v₁‖ · ‖v₂‖)     ∈ [−1, 1]
```

```python
model.cosine_similarity("king", "queen", vocab)   # → ~0.85
model.cosine_similarity("king", "mountain", vocab) # → ~0.10
```

### Nearest Neighbors

```python
model.most_similar("paris", vocab, topn=5)
# → [('berlin', 0.91), ('london', 0.89), ('rome', 0.87), ...]
```

### Word Analogies — Vector Arithmetic

```python
model.analogy("king", "man", "woman", vocab)
# king − man + woman  →  queen

model.analogy("paris", "france", "germany", vocab)
# paris − france + germany  →  berlin
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 50 | Embedding dimensionality |
| `window` | 3 | Max context window size |
| `neg_samples` | 5 | Negative samples per positive pair |
| `epochs` | 300 | Training epochs |
| `lr` | 0.05 | Initial learning rate |
| `lr_min` | 0.0005 | Minimum learning rate |
| `subsample_t` | 1e-3 | Subsampling threshold |
| `min_count` | 1 | Minimum word frequency |

---

## Installation & Usage

```bash
pip install numpy matplotlib scikit-learn
jupyter notebook word2vec_scratch.ipynb
```

Run all cells top to bottom. Training takes ~30 seconds on a laptop CPU.

---

## Key Takeaway

> Skip-Gram with Negative Sampling is logistic regression on word pairs
> using Binary Cross Entropy. Positive pairs pull words together in
> vector space; negative pairs push unrelated words apart. No labels,
> no supervision — just co-occurrence statistics — yet the geometry
> that emerges encodes semantics, syntax, and analogy.

---

## Dependencies

```
numpy
matplotlib
scikit-learn
```

---

*Part 1 of 3 — NLP from Scratch series*  
*Next: BPE Tokenizer from Scratch*
